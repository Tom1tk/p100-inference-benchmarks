#!/usr/bin/env bash
# H41b: 2-card serve command 16k + GGML_PASCAL_MMVH_Q6K_WIDE=1, REPS=2. Control banked: h36-duplex-16k 40.37 / 365.3.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK=/root/llama.cpp-Pascal/build-cuda-p100/bin
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 \
       GGML_CUDA_PASCAL_GEMM_ALGO=1 GGML_PASCAL_DUPLEX_REDUCE=1 GGML_PASCAL_MMVH_Q6K_WIDE=1
export N_PREDICT=400 BATCH=2048 UB=2048 BIN=$FORK/llama-server
REPS=2 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt SPLIT=tensor \
    ./scripts/run-spec-placement.sh h41-wide-16k draft-mtp /root/mtp-Qwen3.8-27B-Q4_0.gguf default
python3 scripts/compare-outputs.py h36-duplex-16k h41-wide-16k || true
grep -E '^(label|h36-duplex-16k|h41-)' results/h11-placement.csv | column -s, -t
git add -A results logs scripts/run-h41b-speed.sh HYPOTHESES.md
git commit -q -m "bench: H41b Q6_K wide mmvh 2-card 16k" && { git push -q && echo pushed || echo "WARNING: push failed"; }
