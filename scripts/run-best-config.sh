#!/usr/bin/env bash
# Confirm the two decode levers stack: -sm tensor + MTP at 16k, with P2P on.
# H10 measured P2P at +2.9% on a drafter-free llama-bench tg128; Phase 2
# measured tensor+MTP at 29.26 t/s at depth. Neither predicts the combination.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

while pgrep -f 'run-buun-tensor.sh|llama-server|llama-bench' >/dev/null; do sleep 30; done
for _ in $(seq 30); do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
    [[ "$t" -le 70 ]] && break
    echo "cooling: hottest ${t}C"; sleep 20
done

export SPLIT=tensor GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 \
       REPS=3 N_PREDICT=400 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt
./scripts/run-spec-placement.sh p2-tensor-mtp-16k-p2p draft-mtp /root/mtp-Qwen3.8-27B-Q4_0.gguf default 2>&1 | tail -8
echo "=== best-config check complete ==="
