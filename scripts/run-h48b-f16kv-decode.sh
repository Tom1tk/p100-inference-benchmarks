#!/usr/bin/env bash
# H48b: f16 KV vs q8_0 98k decode step time. 2 more f16 reps (CACHE_PROMPT=1: one prefill) -> 3 with h48-f16kv-98k.
# Control: p15-200w-98k (2) + p15-200w-98k-r3 (1), q8_0. Metric: ms/step = 1000 * n_decoded / decode_tps / (n_decoded - accepted).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 GGML_CUDA_PASCAL_GEMM_ALGO=1 \
       GGML_PASCAL_DUPLEX_REDUCE=1 GGML_PASCAL_MMVH_Q6K_WIDE=1 GGML_PASCAL_MMVH_LOWBIT=1 GGML_PASCAL_FA_GEMM=1
export N_PREDICT=400 BATCH=2048 UB=2048 BIN=/root/llama.cpp-Pascal/build-cuda-p100/bin/llama-server
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader | paste -sd' ')
echo "cap: $cap"; [[ "$cap" == "200.00 W 200.00 W" ]] || { echo "ERROR: expected 200 W on both cards"; exit 1; }
for _ in $(seq 1 30); do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
    [[ "$t" -le 55 ]] && break; echo "cooling: hottest card ${t}C"; sleep 20
done
CACHE_PROMPT=1 REPS=2 CTX=100000 CTK=f16 CTV=f16 PROMPT_FILE=prompts/h26-100k-prompt.txt NP=1 REQ_TIMEOUT=1800 SPLIT=tensor \
    ./scripts/run-spec-placement.sh h48-f16kv-98k-r23 draft-mtp /root/mtp-Qwen3.8-27B-Q4_0-head96k.gguf default
python3 scripts/compare-outputs.py h48-f16kv-98k h48-f16kv-98k-r23 || true
awk -F, '$16=="ok" && $1 ~ /^(p15-200w-98k(-r3)?|h48-f16kv-98k(-r23)?)$/ {
    g = ($1 ~ /^h48/) ? "f16" : "q8_0"; ms = 1000*$8/$6/($8-$10); s[g]+=ms; n[g]++; printf "%-20s rep%s %6.2f t/s %5.1f%% accept %6.1f ms/step\n", $1, $5, $6, $11, ms }
    END { for (g in s) printf "%s mean %.1f ms/step (n=%d)\n", g, s[g]/n[g], n[g]; printf "f16 vs q8_0: %+.1f%%\n", 100*(s["f16"]/n["f16"])/(s["q8_0"]/n["q8_0"])-100 }' results/h11-placement.csv
cat logs/h48-f16kv-98k-r23.cap 2>/dev/null
git add -A results logs scripts/run-h48b-f16kv-decode.sh HYPOTHESES.md
git commit -q -m "bench: H48b f16 KV 98k decode step time (reps 2-3)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }
