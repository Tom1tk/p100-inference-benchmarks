#!/usr/bin/env bash
# H50: share of the 98k decode step spent in attention. test-backend-ops perf on CUDA0 at per-card decode shapes
# (2 KV heads, GQA 6, D256; nb 1 = plain step, nb 4 = MTP verify), kv 16k/98k, f16/q8_0. Seconds of GPU, no server.
# Attention per step = 16 attention layers x time(nb=4); compare with the measured step (H48/H48b ms/step).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 GGML_CUDA_PASCAL_GEMM_ALGO=1 GGML_PASCAL_FA_GEMM=1
OUT=results/h50-fa-decode.txt
for _ in $(seq 1 30); do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
    [[ "$t" -le 55 ]] && break; echo "cooling: hottest card ${t}C"; sleep 20
done
for rep in 1 2 3; do
    echo "## rep $rep"
    /root/llama.cpp-Pascal/build-cuda-p100/bin/test-backend-ops perf -b CUDA0 -o FLASH_ATTN_EXT \
        -p 'hsk=256,hsv=256,nh=2,nr23=\[6,1\],kv=(16384|98304),nb=(1|4),.*n_kv_max=0' 2>&1 | grep -E 'FLASH_ATTN_EXT\('
done | tee "$OUT"
# us/run per shape, mean of 3 reps, and 16 layers x nb=4 in ms
grep FLASH_ATTN "$OUT" | sed -E 's/.*kv=([0-9]+),nb=([0-9]+),.*type_K=([a-z0-9_]+),.* ([0-9.]+) us\/run.*/\3 kv\1 nb\2 \4/' |
    awk '{k=$1" "$2" "$3; s[k]+=$4; n[k]++} END {for (k in s) {m=s[k]/n[k]; printf "%-22s %8.1f us/call  x16 = %6.2f ms\n", k, m, m*16/1000}}' | sort | tee -a "$OUT"
git add "$OUT" scripts/run-h50-fa-decode.sh
git commit -q -m "bench: H50 decode attention micro-bench" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }
