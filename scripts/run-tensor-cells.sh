#!/usr/bin/env bash
# mainline -sm tensor across the three allreduce backends.
#
# The NCCL leg is already recorded as phase1-mainline-tensor-q4km (exit 134,
# ggml_backend_cuda_comm_allreduce_nccl) -- it is the Linux default and it
# fails on this rig, same root cause as H5. These two cells cover the runtime
# escapes, which need no rebuild. Folds in H10.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODEL=/root/Qwen3.8-27B-UD-Q4_K_M.gguf

for MODE in internal none; do
    echo "=== cooling before $MODE ==="
    for _ in $(seq 30); do
        T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | sort -rn | head -1)
        echo "hottest card: ${T}C"; [[ "$T" -le 70 ]] && break; sleep 20
    done
    echo "=== running: tensor/$MODE ==="
    GGML_CUDA_ALLREDUCE="$MODE" ./scripts/run-bench.sh mainline "$MODEL" tensor \
        "phase1-mainline-tensor-${MODE}-q4km"
    echo "=== tensor/$MODE exit=$? ==="
    sleep 15
done
echo "=== tensor cells complete ==="
