#!/usr/bin/env bash
# H41 follow-up 2: wide vs off KLD at -ub 8 with -sm layer (no tensor-parallel reduce).
# test-backend-ops with fixed inputs gives wide == off to 4 sig figs, yet -sm tensor KLD was 0.0027
# (off-vs-off exact). ~0 here => the kernel is fine and the -sm tensor reduce is timing-sensitive.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK=/root/llama.cpp-Pascal/build-cuda-p100/bin
Q4=/root/Qwen3.8-27B-UD-Q4_K_M.gguf
export GGML_CUDA_PASCAL_MMVH=1
BASE=/root/h28-kld/h41-layer.kld
for W in 0 1; do
    extra=(--kl-divergence-base "$BASE"); [[ $W == 1 ]] && extra+=(--kl-divergence)
    ABORT_TEMP=80 INTERVAL=2 ./scripts/gpu-monitor.sh > "logs/h41-layer-w$W-temps.log" 2>/dev/null & mon=$!
    GGML_PASCAL_MMVH_Q6K_WIDE=$W "$FORK/llama-perplexity" -m "$Q4" -f prompts/h26-100k-prompt.txt -c 2048 --chunks 2 \
        -b 2048 -ub 8 -ngl 99 -fa 1 -sm layer "${extra[@]}" > "logs/h41-layer-w$W.log" 2>&1
    echo "--- layer w$W rc=$?"; kill $mon
done
rm -f "$BASE"
grep -E "Mean    KLD|Maximum KLD|99.9%   KLD|Same top" logs/h41-layer-w1.log
git add logs scripts/run-h41-layer.sh && git commit -qm "bench: H41 wide vs off KLD at ub 8, -sm layer" && git push -q && echo pushed
