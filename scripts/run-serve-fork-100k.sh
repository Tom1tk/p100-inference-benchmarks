#!/usr/bin/env bash
# serve-fork-100k: the adopted fork serve command (README) at 100k, one rep. Fills the README's
# "98k decode: not re-measured" gap (H26's 20.02 t/s is the old stock command) and is the
# control for P5 (attention at depth). Measurement only, no lever. REPS=1: rep 2's prefill
# would hit the prefix cache (H26 known limitation), and decode needs no second sample to
# land in the README. 1-card arm skipped: IQ3_S 1-card at 100k is not a target.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 \
       GGML_CUDA_PASCAL_GEMM_ALGO=1 GGML_PASCAL_DUPLEX_REDUCE=1 \
       SPLIT=tensor REPS=1 N_PREDICT=400 CTX=100000 CTK=q8_0 CTV=q8_0 \
       PROMPT_FILE=prompts/h26-100k-prompt.txt BATCH=2048 UB=2048 NP=1 REQ_TIMEOUT=1800 \
       BIN=/root/llama.cpp-Pascal/build-cuda-p100/bin/llama-server
./scripts/run-spec-placement.sh serve-fork-100k draft-mtp /root/mtp-Qwen3.8-27B-Q4_0.gguf default
grep -E '^(label|h26-100k-mtp-kvq8|serve-fork-100k)' results/h11-placement.csv | column -s, -t
git add -A results logs scripts/run-serve-fork-100k.sh
git commit -q -m "bench: serve-fork-100k (fork serve command at 100k, 1 rep)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }
