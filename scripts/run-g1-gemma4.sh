#!/usr/bin/env bash
# G1: first look at Gemma 4 QAT (Google q4_0 GGUF, unsloth MTP drafter) on card 1 only.
# Single-run baselines, not an A/B: 16k ctx, ~11k-token depth prompt, MTP drafter, fork build.
# Card 0 is left alone (the system-one service lives there).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unset $(env | grep -oE '^GGML_(CUDA_)?PASCAL_[A-Z0-9_]+' || true)
FORK=/root/llama.cpp-Pascal/build-cuda-p100/bin G=/root/gemma4
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader | paste -sd' ')
echo "cap: $cap"; [[ "$cap" == "200.00 W 200.00 W" ]] || { echo "ERROR: expected 200 W on both cards"; exit 1; }
cool() { for _ in $(seq 1 30); do
    t=$(nvidia-smi -i 1 --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    [[ "$t" -le 55 ]] && return; echo "cooling: card 1 ${t}C"; sleep 20; done; }
fin() { git add -A results logs scripts/run-g1-gemma4.sh scripts/run-spec-placement.sh
    git commit -q -m "bench: G1 Gemma 4 on card 1 ($1)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }; }

# downloads run alongside: hf renames a file into place only once it is complete
await() { until [[ -f "$1" && -f "$2" ]]; do sleep 60; done; echo "ready: $(basename "$1")"; }

export N_PREDICT=400 BATCH=2048 UB=2048 BIN=$FORK/llama-server REPS=1 SPLIT=none CVD=1 CTX=16384 \
       PROMPT_FILE=prompts/h11-depth-prompt.txt

await $G/gemma-4-12b-it-qat-q4_0.gguf $G/mtp-gemma-4-12B-it.gguf
cool; MODEL=$G/gemma-4-12b-it-qat-q4_0.gguf ./scripts/run-spec-placement.sh g1-gemma4-12b-mtp-16k draft-mtp $G/mtp-gemma-4-12B-it.gguf default
fin 12b
await $G/gemma-4-E4B_q4_0-it.gguf $G/mtp-gemma-4-E4B-it.gguf
cool; MODEL=$G/gemma-4-E4B_q4_0-it.gguf ./scripts/run-spec-placement.sh g1-gemma4-e4b-mtp-16k draft-mtp $G/mtp-gemma-4-E4B-it.gguf default
fin e4b
echo "=== G1 done ==="
