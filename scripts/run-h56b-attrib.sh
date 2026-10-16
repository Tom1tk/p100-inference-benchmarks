#!/usr/bin/env bash
# H56b: attribute the H56 text change. Rebased fork WITHOUT the tidy commits (59acd5b, worktree /root/lcp-h56b),
# every lever set =1 explicitly as in r10s. Output == h56-rebase-16k -> upstream changed it, tidy is neutral.
# Output == r10s-fork-16k -> the tidy changed it. REPS=3, 2-card 16k serve command.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader | paste -sd' ')
echo "cap: $cap"; [[ "$cap" == "200.00 W 200.00 W" ]] || { echo "ERROR: expected 200 W on both cards"; exit 1; }
export GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 GGML_CUDA_PASCAL_GEMM_ALGO=1 GGML_PASCAL_DUPLEX_REDUCE=1 \
    GGML_PASCAL_MMVH_Q6K_WIDE=1 GGML_PASCAL_MMVH_LOWBIT=1 GGML_PASCAL_FA_GEMM=1
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1
export N_PREDICT=400 BATCH=2048 UB=2048 BIN=/root/lcp-h56b/build/bin/llama-server
env | grep -E '^GGML_' | sort
for _ in $(seq 1 30); do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
    [[ "$t" -le 55 ]] && break; echo "cooling: hottest card ${t}C"; sleep 20
done
REPS=3 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt SPLIT=tensor \
    ./scripts/run-spec-placement.sh h56b-pretidy-16k draft-mtp /root/mtp-Qwen3.8-27B-Q4_0-head96k.gguf default
python3 scripts/compare-outputs.py h56-rebase-16k h56b-pretidy-16k r10s-fork-16k h56b-pretidy-16k
grep -E '^h56b-' results/h11-placement.csv | cut -d, -f1,5-7,16 | column -s, -t
git add -A results logs scripts/run-h56b-attrib.sh
git commit -q -m "bench: H56b attribution (pre-tidy rebased build)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }
