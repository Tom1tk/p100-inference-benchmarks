#!/usr/bin/env bash
# H39r 2-card follow-up: GGML_PASCAL_DEQUANT_STAGED=1 on the 2-card serve command, 16k, REPS=3, 200 W.
# Control banked at the same cap and settings: p15-200w-16k (2 reps) + p15-200w-16k-r3 (1 rep).
# Pass: prefill >= 2% over the control mean, output identical.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 GGML_CUDA_PASCAL_GEMM_ALGO=1 \
       GGML_PASCAL_DUPLEX_REDUCE=1 GGML_PASCAL_MMVH_Q6K_WIDE=1 GGML_PASCAL_MMVH_LOWBIT=1 GGML_PASCAL_FA_GEMM=1 GGML_PASCAL_DEQUANT_STAGED=1
export N_PREDICT=400 BATCH=2048 UB=2048 BIN=/root/llama.cpp-Pascal/build-cuda-p100/bin/llama-server
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader | paste -sd' ')
[[ "$cap" == "200.00 W 200.00 W" ]] || { echo "ERROR: expected 200 W on both cards"; exit 1; }
REPS=3 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt SPLIT=tensor \
    ./scripts/run-spec-placement.sh h39r-2card-staged draft-mtp /root/mtp-Qwen3.8-27B-Q4_0-head96k.gguf default
python3 scripts/compare-outputs.py p15-200w-16k h39r-2card-staged || echo "WARNING: output differs"
pm() { awk -F, -v re="$1" '$1 ~ re && $16=="ok" {s+=$7; n++} END {if (n) printf "%.2f", s/n; else print 0}' results/h11-placement.csv; }
C=$(pm '^p15-200w-16k(-r3)?$'); S=$(pm '^h39r-2card-staged$')
echo "2-card 16k prefill: control $C, staged $S ($(python3 -c "print(f'{100*($S/$C-1):+.1f}%')"))"
git add -A results logs scripts/run-h39r-2card.sh
git commit -q -m "bench: H39r staged dequant 2-card at 200 W" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }
