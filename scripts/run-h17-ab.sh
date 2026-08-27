#!/usr/bin/env bash
# H17: paired A/B of the sm_60 FP16 fast-path patch.
# Same tree, same flags, same session -- the only difference is the build.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL=/root/Qwen3.8-27B-UD-Q4_K_M.gguf
export GGML_CUDA_ALLREDUCE=none
export REPS=2 PROMPT_DEPTHS="2048,16384" GEN_TOKENS=128

echo "=== ARM 1/2: unpatched control (mainline-rebased) ==="
./scripts/run-bench.sh mainline-rebased "$MODEL" tensor h17-control-nofix -b 2048 -ub 512

echo
echo "=== ARM 2/2: patched (rebased-h17, sm_60 out of the FP16 fast path) ==="
./scripts/run-bench.sh rebased-h17 "$MODEL" tensor h17-patched-fp32 -b 2048 -ub 512

echo
echo "=== A/B done -- compare results/raw/h17-control-nofix.csv vs h17-patched-fp32.csv ==="
