#!/usr/bin/env bash
# Re-run of the tensor+MTP+P2P cell. The first attempt died because
# run-spec-placement.sh was edited while that job was executing it -- bash
# reads scripts incrementally, so the byte offsets shifted mid-run and it
# fell over with a syntax error. Harness defect, no data lost.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

while pgrep -f 'run-buun-turbo.sh|llama-server|llama-bench' >/dev/null; do sleep 30; done
for _ in $(seq 30); do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
    [[ "$t" -le 70 ]] && break
    echo "cooling: hottest ${t}C"; sleep 20
done

export SPLIT=tensor GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 \
       REPS=3 N_PREDICT=400 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt
./scripts/run-spec-placement.sh p2-tensor-mtp-16k-p2p draft-mtp /root/mtp-Qwen3.8-27B-Q4_0.gguf default 2>&1 | tail -8
echo "=== best-config rerun complete ==="
