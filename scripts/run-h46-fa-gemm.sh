#!/usr/bin/env bash
# H46: GGML_PASCAL_FA_GEMM=1, prefill flash attention through cuBLAS (fork fattn-pascal-gemm.cuh).
# Micro-bench (test-backend-ops perf, D256, 2 kv heads, GQA 6, nb 2048, q8_0 KV): 5.54 -> 8.57 TFLOPS at kv 16k and 64k.
# 1) KLD gate at depth, H38 style: Q4_K_M 2-card, -c 8192 --chunks 2, q8_0 KV, on vs off.
#    Pass = mean KLD <= 0.001 and same-top >= 99.5%.
# 2) 1-card IQ3_S 16k, REPS=3. Control banked: h42-1card-iq3-lowbit 17.89 / 201.2 (later levers are decode-only).
# 3) 2-card serve command at 100k: fresh control REPS=3 (serve-fork-100k is 1 rep and predates H41-H45),
#    then FA_GEMM REPS=3. The GEMM arm stops after rep 1 if prefill is not 3% over the control mean.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK=/root/llama.cpp-Pascal/build-cuda-p100/bin
Q4=/root/Qwen3.8-27B-UD-Q4_K_M.gguf
IQ3=/root/Qwen3.8-27B-UD-IQ3_S.gguf
MTP=/root/mtp-Qwen3.8-27B-Q4_0-head96k.gguf
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 GGML_CUDA_PASCAL_GEMM_ALGO=1 \
       GGML_PASCAL_DUPLEX_REDUCE=1 GGML_PASCAL_MMVH_Q6K_WIDE=1 GGML_PASCAL_MMVH_LOWBIT=1
cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 60 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20
    done
}
ppl() {  # label fa_gemm extra-args...
    cool
    ABORT_TEMP=80 INTERVAL=2 ./scripts/gpu-monitor.sh > "logs/h46-$1-temps.log" 2>/dev/null &
    local mon=$!
    GGML_PASCAL_FA_GEMM=$2 "$FORK/llama-perplexity" -m "$Q4" -f prompts/h26-100k-prompt.txt -c 8192 --chunks 2 \
        -b 2048 -ub 2048 -ngl 99 -fa 1 -sm tensor -ctk q8_0 -ctv q8_0 "${@:3}" > "logs/h46-$1.log" 2>&1
    local rc=$?; kill $mon
    echo "--- $1 rc=$rc, peak $(tr '|' '\n' < "logs/h46-$1-temps.log" | awk -F', ' 'NF>=3{print $2}' | sort -n | tail -1)C, $(grep -m1 -E 'Final estimate|Mean    KLD' "logs/h46-$1.log") ---"
    return $rc
}
fin() { git add -A results logs scripts/run-h46-fa-gemm.sh HYPOTHESES.md
    git commit -q -m "bench: H46 GEMM prefill FA ($1)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }; }
prefill_mean() {  # label -> mean prefill_tps of ok rows
    awk -F, -v l="$1" '$1==l && $16=="ok" {s+=$7; n++} END {if (n) printf "%.2f", s/n; else print 0}' results/h11-placement.csv
}

BASE=/root/h28-kld/h46-base.kld; mkdir -p /root/h28-kld
ppl off 0 --kl-divergence-base "$BASE" || { rm -f "$BASE"; fin "perplexity failed"; exit 1; }
ppl gemm 1 --kl-divergence-base "$BASE" --kl-divergence || { rm -f "$BASE"; fin "perplexity failed"; exit 1; }
rm -f "$BASE"
K=$(grep -m1 'Mean    KLD:' logs/h46-gemm.log | awk '{print $3}')
T=$(grep -m1 'Same top p:' logs/h46-gemm.log | awk '{print $4}')
echo "gemm vs off: mean KLD $K, same top $T"
python3 -c "import sys; sys.exit(0 if float('$K') <= 0.001 and float('$T') >= 99.5 else 1)" \
    || { echo "gate: KLD FAIL"; fin "KLD gate FAILED"; exit 1; }
echo "gate: KLD PASS"
fin "KLD gate"

export N_PREDICT=400 BATCH=2048 UB=2048 BIN=$FORK/llama-server
cool
GGML_PASCAL_FA_GEMM=1 REPS=3 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt SPLIT=none MODEL="$IQ3" \
    ./scripts/run-spec-placement.sh h46-1card-iq3-fagemm none none default
fin "1-card 16k"

run100k() {  # label fa_gemm reps
    cool
    GGML_PASCAL_FA_GEMM=$2 REPS=$3 CTX=100000 CTK=q8_0 CTV=q8_0 PROMPT_FILE=prompts/h26-100k-prompt.txt NP=1 \
        REQ_TIMEOUT=1800 SPLIT=tensor ./scripts/run-spec-placement.sh "$1" draft-mtp "$MTP" default
}
run100k h46-ctrl-100k 0 3
fin "2-card 100k control"
CTRL=$(prefill_mean h46-ctrl-100k)
run100k h46-fagemm-100k 1 1
G1=$(prefill_mean h46-fagemm-100k)
echo "100k prefill: control mean $CTRL, fagemm rep 1 $G1"
if python3 -c "import sys; sys.exit(0 if float('$G1') > 1.03*float('$CTRL') else 1)"; then
    run100k h46-fagemm-100k-r23 1 2
else
    echo "skip: fagemm rep 1 not 3% over control, reps 2-3 cancelled"
fi
python3 scripts/compare-outputs.py h46-ctrl-100k h46-fagemm-100k || true
grep -E '^(label|h42-1card-iq3-lowbit|serve-fork-100k|h46-)' results/h11-placement.csv | column -s, -t
fin "2-card 100k"
