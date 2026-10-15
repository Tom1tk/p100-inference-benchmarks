#!/usr/bin/env bash
# H46 end-to-end gate: Phase 8 tier 2 100k haystack, current serve env + GGML_PASCAL_FA_GEMM=1.
# Pass = answers every question p8-ref (Q5_K_M reference, 18/19) answers. Reference banked, not re-run.
# Drafter is the script's default; greedy speculative output does not depend on it.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 GGML_CUDA_PASCAL_GEMM_ALGO=1 \
       GGML_PASCAL_DUPLEX_REDUCE=1 GGML_PASCAL_MMVH_Q6K_WIDE=1 GGML_PASCAL_MMVH_LOWBIT=1 GGML_PASCAL_FA_GEMM=1
BIN=/root/llama.cpp-Pascal/build-cuda-p100/bin/llama-server MODEL=/root/Qwen3.8-27B-UD-Q4_K_M.gguf SPEC=mtp \
    ./scripts/run-phase8-tier2.sh p8-h46-fagemm
awk -F, '$1=="p8-ref"||$1=="p8-candidate"||$1=="p8-h46-fagemm"{n[$1]++; h[$1]+=$5} END{for(l in n) print l, h[l]"/"n[l]}' results/phase8-tier2.csv
awk -F, '$1=="p8-ref"&&$5==1{r[$2","$3]=1} $1=="p8-h46-fagemm"&&$5!=1&&r[$2","$3]{print "MISS vs ref:", $2, $3}' results/phase8-tier2.csv
git add -A results logs scripts/run-h46-haystack.sh
git commit -q -m "bench: H46 100k haystack with FA_GEMM" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }
