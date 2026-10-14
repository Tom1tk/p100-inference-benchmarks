#!/usr/bin/env bash
# H38 (fork plan P5): half2 Q.K accumulation in the fattn-tile kernel (GGML_PASCAL_FA_KQ_H2, compile-time,
# build-p5). sm60 KQ inner loop was hmul2 + 2 cvt + 2 fp32 adds per half2; now cpy_ne __hfma2 then one flush.
# Micro-bench (P5 perf case, D256, GQA 6, kv 32k, nb 512): 37.4 -> 27.8 ms (-26%). Lossy (fp16 partials).
# 1) KLD gate at depth: -c 8192 --chunks 2 (scored tokens sit 4-8k deep), serve levers, f16 KV.
#    Pass = mean KLD <= 0.001 (<2% of the Q4 quant step, H30 0.057) and same-top >= 99.5%.
# 2) 1-card IQ3_S at 16k, drafter-free, REPS=2. Control banked: h34-1card-iq3 rep 2, 15.83 / 196.72.
# 3) 2-card serve command at 100k, REPS=1. Control banked: serve-fork-100k 26.94 / 231.97.
# Lossy at depth: if adopted, the H32 100k haystack is re-run before the README command changes.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN=/root/llama.cpp-Pascal/build-cuda-p100/bin
H2=/root/llama.cpp-Pascal/build-p5/bin
Q4=/root/Qwen3.8-27B-UD-Q4_K_M.gguf
IQ3=/root/Qwen3.8-27B-UD-IQ3_S.gguf
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 \
       GGML_CUDA_PASCAL_GEMM_ALGO=1 GGML_PASCAL_DUPLEX_REDUCE=1
cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 60 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20
    done
}
ppl() {  # label bin extra-args...
    cool
    ABORT_TEMP=80 INTERVAL=2 ./scripts/gpu-monitor.sh > "logs/h38-$1-temps.log" 2>/dev/null &
    local mon=$!
    "$2/llama-perplexity" -m "$Q4" -f prompts/h26-100k-prompt.txt -c 8192 --chunks 2 -b 2048 -ub 2048 \
        -ngl 99 -fa 1 -sm tensor "${@:3}" > "logs/h38-$1.log" 2>&1
    local rc=$?; kill $mon
    echo "--- $1 rc=$rc, peak $(tr '|' '\n' < "logs/h38-$1-temps.log" | awk -F', ' 'NF>=3{print $2}' | sort -n | tail -1)C, $(grep -m1 -E 'Final estimate|Mean    KLD' "logs/h38-$1.log") ---"
    return $rc
}
fin() { git add -A results logs scripts/run-h38-fa-kq-h2.sh
    git commit -q -m "bench: H38 FA half2 KQ accumulation ($1)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }; }

BASE=/root/h28-kld/h38-base.kld; mkdir -p /root/h28-kld
ppl kq-f32 "$MAIN" --kl-divergence-base "$BASE" || { rm -f "$BASE"; fin "perplexity failed"; exit 1; }
ppl kq-h2 "$H2" --kl-divergence-base "$BASE" --kl-divergence || { rm -f "$BASE"; fin "perplexity failed"; exit 1; }
rm -f "$BASE"
K=$(grep -m1 'Mean    KLD:' logs/h38-kq-h2.log | awk '{print $3}')
T=$(grep -m1 'Same top p:' logs/h38-kq-h2.log | awk '{print $4}')
echo "kq-h2 vs kq-f32: mean KLD $K, same top $T"
python3 -c "import sys; sys.exit(0 if float('$K') <= 0.001 and float('$T') >= 99.5 else 1)" \
    || { echo "gate: KLD FAIL"; fin "KLD gate FAILED"; exit 1; }
echo "gate: KLD PASS"

export N_PREDICT=400 BATCH=2048 UB=2048 BIN=$H2/llama-server
cool
REPS=2 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt SPLIT=none MODEL="$IQ3" \
    ./scripts/run-spec-placement.sh h38-1card-iq3-kqh2 none none default
cool
REPS=1 CTX=100000 CTK=q8_0 CTV=q8_0 PROMPT_FILE=prompts/h26-100k-prompt.txt NP=1 REQ_TIMEOUT=1800 SPLIT=tensor \
    ./scripts/run-spec-placement.sh h38-kqh2-100k draft-mtp /root/mtp-Qwen3.8-27B-Q4_0.gguf default
grep -E '^(label|h34-1card-iq3|serve-fork-100k|h38-)' results/h11-placement.csv | column -s, -t
fin "KLD gate + 1-card 16k + 2-card 100k"
