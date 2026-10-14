#!/usr/bin/env bash
# Confirmation run of the adopted serve command (H32 pass) at 16k, so README carries measured
# numbers. Same harness as banked h28-2card-mmvh1-mtp-p2p (37.75 / 320.25, rep 2), plus the
# three levers adopted since (Q6_K mmvh is inside MMVH=1; F16 reduce; ALGO3).
# Prediction: decode +3-6% (Q6_K, H34), prefill +8-10% (F16 reduce + ALGO3, H34/H35).
# 1-card arm skipped (rule zero): H34/H35 already measured 1-card with every lever that applies.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 60 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20
    done
}
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 \
       GGML_CUDA_PASCAL_GEMM_ALGO=1 REPS=2 N_PREDICT=400 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt \
       BATCH=2048 UB=2048 BIN=/root/llama.cpp-Pascal/build-cuda-p100/bin/llama-server
cool
SPLIT=tensor ./scripts/run-spec-placement.sh serve-fork-16k draft-mtp /root/mtp-Qwen3.8-27B-Q4_0.gguf default
grep -E '^(label|h28-2card-mmvh1-mtp-p2p|serve-fork-16k)' results/h11-placement.csv | column -s, -t
git add -A results logs scripts/run-serve-confirm-16k.sh
git commit -q -m "bench: adopted serve command, 16k confirmation (MTP + all fork levers)" \
    && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }
