#!/usr/bin/env bash
# H32 (Phase 8 tier 2): reference, then the candidate serve command. Commit + push after each.
# p8-1card skipped (pre-registered): IQ3_S does not hold 100k ctx on one 16 GB card (H20/H25).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1
push() { git add -A results logs scripts/run-h32-tier2.sh
         git commit -q -m "bench: H32 Phase 8 tier 2 — $1" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }; }
BIN=/root/llama.cpp-Pascal/build-h17/bin/llama-server MODEL=/root/Qwen3.8-27B-UD-Q5_K_M.gguf SPEC=none \
    ./scripts/run-phase8-tier2.sh p8-ref
push p8-ref
sleep 60
BIN=/root/llama.cpp-Pascal/build-cuda-p100/bin/llama-server MODEL=/root/Qwen3.8-27B-UD-Q4_K_M.gguf SPEC=mtp \
GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 GGML_CUDA_PASCAL_GEMM_ALGO=1 \
    ./scripts/run-phase8-tier2.sh p8-candidate
push p8-candidate
tail -n 45 results/phase8-tier2.csv 2>/dev/null | cut -c1-160
