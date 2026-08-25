#!/usr/bin/env bash
# Does a drafter load at all under -sm tensor? Cheap gate before building a matrix.
#
# Prediction (from H11): MTP ships its own output.weight + token_embd.weight and
# should be fine. DFlash2 ships NEITHER and borrows the target's via ctx_other --
# under -sm tensor the target's output.weight is SPLIT ACROSS BOTH CARDS, which
# is a harder case than the -sm layer one that already aborted on -devd CUDA0.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export SPLIT=tensor GGML_CUDA_ALLREDUCE=internal REPS=1 N_PREDICT=64

run() {
    echo "########## $1"
    ./scripts/run-spec-placement.sh "$@" 2>&1 | tail -6
    echo "---- exit=$? ----"
}
run smoke-tensor-none    none    none                                     default
run smoke-tensor-mtp     draft-mtp    /root/mtp-Qwen3.8-27B-Q4_0.gguf       default
run smoke-tensor-df2q4   draft-dflash /root/Qwen3.8-27B-DFlash2-Q4_K_M.gguf default
echo "=== smoke complete ==="
