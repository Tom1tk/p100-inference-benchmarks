#!/usr/bin/env bash
# H41 follow-up: off-vs-off KLD at -ub 8 (same config as run-h41-q6k-wide.sh step 1) to get the
# run-to-run noise floor. If off-vs-off shows the same tail (max KLD > 0.5 or same top < 99.9) the
# H41 gate failure is nondeterminism, not the wide kernel.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK=/root/llama.cpp-Pascal/build-cuda-p100/bin
Q4=/root/Qwen3.8-27B-UD-Q4_K_M.gguf
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 \
       GGML_CUDA_PASCAL_GEMM_ALGO=1 GGML_PASCAL_DUPLEX_REDUCE=1 GGML_PASCAL_MMVH_Q6K_WIDE=0
BASE=/root/h28-kld/h41-floor.kld
for arm in floor-base floor-off; do
    extra=(--kl-divergence-base "$BASE"); [[ $arm == floor-off ]] && extra+=(--kl-divergence)
    ABORT_TEMP=80 INTERVAL=2 ./scripts/gpu-monitor.sh > "logs/h41-$arm-temps.log" 2>/dev/null & mon=$!
    "$FORK/llama-perplexity" -m "$Q4" -f prompts/h26-100k-prompt.txt -c 2048 --chunks 2 \
        -b 2048 -ub 8 -ngl 99 -fa 1 -sm tensor "${extra[@]}" > "logs/h41-$arm.log" 2>&1
    echo "--- $arm rc=$?"; kill $mon
done
rm -f "$BASE"
grep -E "Mean    KLD|Maximum KLD|99.9%   KLD|Same top" logs/h41-floor-off.log
git add logs scripts/run-h41-floor.sh && git commit -qm "bench: H41 off-vs-off KLD floor at ub 8" && git push -q && echo pushed
