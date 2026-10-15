#!/usr/bin/env bash
# H46 diagnosis after the KLD gate failed (0.320): bisect the GEMM FA path's precision loss.
# KLD vs FA_GEMM=0 at -c 8192 --chunks 2 (same setup as run-h46-fa-gemm.sh); base kept for later arms.
# usage: [BASE=file] [PPL_ARGS="-fa 1 ..."] run-h46-diag.sh label ENV=VAL...   (label "base*" writes the base file)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK=/root/llama.cpp-Pascal/build-cuda-p100/bin
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 GGML_CUDA_PASCAL_GEMM_ALGO=1 \
       GGML_PASCAL_DUPLEX_REDUCE=1 GGML_PASCAL_MMVH_Q6K_WIDE=1 GGML_PASCAL_MMVH_LOWBIT=1
BASE=${BASE:-/root/h28-kld/h46-diag-base.kld}
L=$1; shift
if [[ $L == base* ]]; then KARGS=(--kl-divergence-base "$BASE"); else KARGS=(--kl-divergence-base "$BASE" --kl-divergence); fi
ABORT_TEMP=80 INTERVAL=2 ./scripts/gpu-monitor.sh > "logs/h46-diag-$L-temps.log" 2>/dev/null &
mon=$!
env "$@" "$FORK/llama-perplexity" -m /root/Qwen3.8-27B-UD-Q4_K_M.gguf -f prompts/h26-100k-prompt.txt -c 8192 --chunks 2 \
    -b 2048 -ub 2048 -ngl 99 ${PPL_ARGS:--sm tensor -fa 1 -ctk q8_0 -ctv q8_0} "${KARGS[@]}" > "logs/h46-diag-$L.log" 2>&1
rc=$?; kill $mon
echo "--- $L ($* | ${PPL_ARGS:-})  rc=$rc, $(grep -m1 -E 'Final estimate|Mean    KLD' "logs/h46-diag-$L.log"), $(grep -m1 'Same top p:' "logs/h46-diag-$L.log") ---"
git add -A logs scripts/run-h46-diag.sh && git commit -q -m "bench: H46 diag $L ($*)" && git push -q
