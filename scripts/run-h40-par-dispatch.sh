#!/usr/bin/env bash
# H40 (fork plan D-new): GGML_PASCAL_PAR_DISPATCH=1, a host dispatch thread per extra card in the meta
# backend, so both cards' subgraphs are issued at once. P2 step-0 trace: decode is host-bound, ~9 us host
# per launch vs ~20 us GPU per kernel, cards issued serially. Bit-identical by construction.
# 1) Identity: Q4_K_M 2-card, -c 2048 --chunks 2, KLD on vs off. Pass = |KLD| <= 1e-5, same top 100%.
# 2) 2-card serve command 16k, REPS=2. Control banked: h36-duplex-16k, 40.37 / 365.3.
#    No 1-card arm: -sm none never enters the meta backend (rule zero).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK=/root/llama.cpp-Pascal/build-cuda-p100/bin
Q4=/root/Qwen3.8-27B-UD-Q4_K_M.gguf
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 \
       GGML_CUDA_PASCAL_GEMM_ALGO=1 GGML_PASCAL_DUPLEX_REDUCE=1
cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 60 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20
    done
}
ppl() {  # label par-on extra-args...
    cool
    ABORT_TEMP=80 INTERVAL=2 ./scripts/gpu-monitor.sh > "logs/h40-$1-temps.log" 2>/dev/null &
    local mon=$!
    GGML_PASCAL_PAR_DISPATCH=$2 "$FORK/llama-perplexity" -m "$Q4" -f prompts/h26-100k-prompt.txt -c 2048 --chunks 2 \
        -b 2048 -ub 2048 -ngl 99 -fa 1 -sm tensor "${@:3}" > "logs/h40-$1.log" 2>&1
    local rc=$?; kill $mon
    echo "--- $1 rc=$rc, peak $(tr '|' '\n' < "logs/h40-$1-temps.log" | awk -F', ' 'NF>=3{print $2}' | sort -n | tail -1)C, $(grep -m1 -E 'Final estimate|Mean    KLD' "logs/h40-$1.log") ---"
    return $rc
}
fin() { git add -A results logs scripts/run-h40-par-dispatch.sh
    git commit -q -m "bench: H40 parallel dispatch ($1)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }; }

BASE=/root/h28-kld/h40-base.kld; mkdir -p /root/h28-kld
ppl off 0 --kl-divergence-base "$BASE" || { rm -f "$BASE"; fin "perplexity failed"; exit 1; }
ppl par 1 --kl-divergence-base "$BASE" --kl-divergence || { rm -f "$BASE"; fin "perplexity failed"; exit 1; }
rm -f "$BASE"
K=$(grep -m1 'Mean    KLD:' logs/h40-par.log | awk '{print $3}')
T=$(grep -m1 'Same top p:' logs/h40-par.log | awk '{print $4}')
echo "par vs off: mean KLD $K, same top $T"
python3 -c "import sys; sys.exit(0 if abs(float('$K')) <= 1e-5 and float('$T') >= 100 else 1)" \
    || { echo "gate: identity FAIL"; fin "identity FAILED"; exit 1; }
echo "gate: identity PASS"

export GGML_PASCAL_PAR_DISPATCH=1 N_PREDICT=400 BATCH=2048 UB=2048 BIN=$FORK/llama-server
cool
REPS=2 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt SPLIT=tensor \
    ./scripts/run-spec-placement.sh h40-par-16k draft-mtp /root/mtp-Qwen3.8-27B-Q4_0.gguf default
python3 scripts/compare-outputs.py h36-duplex-16k h40-par-16k  # decode-path identity
grep -E '^(label|h36-duplex-16k|h40-)' results/h11-placement.csv | column -s, -t
fin "identity + 2-card 16k"
