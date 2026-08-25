#!/usr/bin/env bash
# Phase 2: speculative decoding under -sm tensor (rebased build).
#
# The smoke test showed MTP + tensor at 32.60 t/s vs 20.13 control, but at
# N_PREDICT=64 with 90% acceptance -- far above the 44.9% the same drafter
# gets at N_PREDICT=400 on layer split. Short generations flatter the drafter,
# so this matrix re-runs at N_PREDICT=400 / REPS=3, matched cell-for-cell
# against the h11-* layer rows so the tensor-vs-layer delta is attributable.
#
# DFlash2 is absent by necessity: it aborts under tensor split in the
# meta-backend split planner (ggml-backend-meta.cpp:543, per-row op fed an
# axis-0-split source -- the borrowed target output.weight). Layer-only.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export SPLIT=tensor GGML_CUDA_ALLREDUCE=internal REPS=3 N_PREDICT=400

cool() {
    for _ in $(seq 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 70 ]] && return 0
        echo "cooling: hottest ${t}C"; sleep 20
    done
}

run() {
    cool
    echo "########## $1"
    ./scripts/run-spec-placement.sh "$@" 2>&1 | tail -8
}

# --- 4k: like-for-like with h11-none-control / h11-mtp-default --------------
CTX=4096 run p2-tensor-none none      none                            default
CTX=4096 run p2-tensor-mtp  draft-mtp /root/mtp-Qwen3.8-27B-Q4_0.gguf default

# --- 16k: like-for-like with h11-depth16k-* --------------------------------
export CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt
run p2-tensor-none-16k none      none                            default
run p2-tensor-mtp-16k  draft-mtp /root/mtp-Qwen3.8-27B-Q4_0.gguf default

echo "=== phase2 tensor matrix complete ==="
