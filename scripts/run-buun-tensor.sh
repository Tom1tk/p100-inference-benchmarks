#!/usr/bin/env bash
# buun is the only tree that has all three of -sm tensor, draft-mtp, and
# TurboQuant KV. mainline/rebased has tensor+MTP but no turbo types; ik has
# turbo... no, ik has neither turbo nor tensor. So if buun's tensor split
# performs like mainline's, buun becomes the engine for the target config.
#
# Step 1 is just: does buun -sm tensor match rebased mainline -sm tensor?
# Baseline is 222.75 / 223.43 / 214.88 / 208.25 / 20.34 (h10-p2p-off-q4km).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

while pgrep -f 'run-h10-p2p.sh|run-layer-16k.sh|llama-server|llama-bench' >/dev/null; do sleep 30; done

for _ in $(seq 30); do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
    [[ "$t" -le 70 ]] && break
    echo "cooling: hottest ${t}C"; sleep 20
done

GGML_CUDA_ALLREDUCE=none ./scripts/run-bench.sh buun \
    /root/Qwen3.8-27B-UD-Q4_K_M.gguf tensor phase1-buun-tensor-q4km 2>&1 | tail -14

echo "=== buun tensor complete ==="
