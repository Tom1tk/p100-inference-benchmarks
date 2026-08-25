#!/usr/bin/env bash
# Does the rebase change throughput? Same model, same flags, same split modes
# as the pre-rebase cells -- the only variable is 115 upstream commits.
# Targets d9b6be07 (cuBLAS static workspace), which touches our only prefill path.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODEL=/root/Qwen3.8-27B-UD-Q4_K_M.gguf

for SPLIT in tensor layer; do
    echo "=== cooling before $SPLIT ==="
    for _ in $(seq 30); do
        T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | sort -rn | head -1)
        echo "hottest card: ${T}C"; [[ "$T" -le 70 ]] && break; sleep 20
    done
    echo "=== running: rebased/$SPLIT ==="
    GGML_CUDA_ALLREDUCE=internal ./scripts/run-bench.sh mainline-rebased "$MODEL" "$SPLIT" \
        "phase1-rebased-${SPLIT}-q4km"
    echo "=== rebased/$SPLIT exit=$? ==="
    sleep 15
done
echo "=== rebase check complete ==="
