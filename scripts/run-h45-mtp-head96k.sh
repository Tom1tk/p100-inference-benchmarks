#!/usr/bin/env bash
# H45: MTP drafter LM head trimmed to the first 98,304 vocab rows (fork scripts/pascal-trim-mtp-head.py).
# 2-card 16k, REPS=3, serve env. Control banked: h44-lowbit-16k 44.96 / 366.4, acceptance 72.8%.
# Gate: output identical, decode up, acceptance within about 1 point. 1-card arm parked (no drafter there).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 GGML_CUDA_PASCAL_GEMM_ALGO=1 \
       GGML_PASCAL_DUPLEX_REDUCE=1 GGML_PASCAL_MMVH_Q6K_WIDE=1 GGML_PASCAL_MMVH_LOWBIT=1
export N_PREDICT=400 BATCH=2048 UB=2048 BIN=/root/llama.cpp-Pascal/build-cuda-p100/bin/llama-server
REPS=3 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt SPLIT=tensor \
    ./scripts/run-spec-placement.sh h45-head96k-16k draft-mtp /root/mtp-Qwen3.8-27B-Q4_0-head96k.gguf default
python3 scripts/compare-outputs.py h44-lowbit-16k h45-head96k-16k || true
grep -E '^(label|h44-lowbit-16k|h45-)' results/h11-placement.csv | column -s, -t
git add -A results logs scripts/run-h45-mtp-head96k.sh HYPOTHESES.md
git commit -q -m "bench: H45 MTP head trimmed to 96k vocab, 2-card 16k" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }
