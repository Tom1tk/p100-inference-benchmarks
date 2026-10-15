#!/usr/bin/env bash
# H43: serve command 16k on fork aea9046 (branch-free Q4_K/Q5_K scales), REPS=2. Control banked: h41-wide-16k 41.51 / 365.4.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK=/root/llama.cpp-Pascal/build-cuda-p100/bin
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 \
       GGML_CUDA_PASCAL_GEMM_ALGO=1 GGML_PASCAL_DUPLEX_REDUCE=1 GGML_PASCAL_MMVH_Q6K_WIDE=1
export N_PREDICT=400 BATCH=2048 UB=2048 BIN=$FORK/llama-server
REPS=2 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt SPLIT=tensor \
    ./scripts/run-spec-placement.sh h43-k4scale-16k draft-mtp /root/mtp-Qwen3.8-27B-Q4_0.gguf default
python3 scripts/compare-outputs.py h41-wide-16k h43-k4scale-16k || true
grep -E '^(label|h41-wide-16k|h43-)' results/h11-placement.csv | column -s, -t
git add -A results logs scripts/run-h43-k4scale.sh HYPOTHESES.md
git commit -q -m "bench: H43 branch-free K-quant scales 2-card 16k" && { git push -q && echo pushed || echo "WARNING: push failed"; }
