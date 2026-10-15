#!/usr/bin/env bash
# Phase 5: the agentic web build on the adopted serve command (README "The current answer"), 2 cards, index 1 (index 0 = aborted p5-fork-serve).
# Harness: opencode v2 (user call 2026-10-15; replaces pi). Also measures H51's baseline cache hit rate (results/web/<label>.json).
# One arm only (rule zero): a stock or per-lever web run cannot change the command. No 1-card arm:
# IQ3_S can't hold 100k ctx on one card (H20/H25), so it isn't the deployed config.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 GGML_CUDA_PASCAL_GEMM_ALGO=1 \
       GGML_PASCAL_DUPLEX_REDUCE=1 GGML_PASCAL_MMVH_Q6K_WIDE=1 GGML_PASCAL_MMVH_LOWBIT=1 GGML_PASCAL_FA_GEMM=1
CTK=f16 CTV=f16 CTX=100000 ./scripts/run-web-bench.sh fork /root/Qwen3.8-27B-UD-Q4_K_M.gguf tensor p5-fork-opencode 1 \
    --spec-type draft-mtp -md /root/mtp-Qwen3.8-27B-Q4_0-head96k.gguf -b 2048 -ub 2048 -np 1
