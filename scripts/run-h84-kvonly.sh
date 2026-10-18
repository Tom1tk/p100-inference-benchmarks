#!/usr/bin/env bash
# H84: KV-only MTP catch-up (GGML_PASCAL_MTP_KV_ONLY). One arm, the build default:
# 16k serve REPS=3 vs banked h83-splitstate-16k (same build minus the port); output must be byte-identical,
# haystack only if it differs.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unset $(env | grep -oE '^GGML_(CUDA_)?PASCAL_[A-Z0-9_]+' || true)
FORK=/root/llama.cpp-Pascal/build-cuda-p100/bin MTP=/root/mtp-Qwen3.8-27B-Q4_0-head96k.gguf
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader | paste -sd' ')
echo "cap: $cap"; [[ "$cap" == "200.00 W 200.00 W" ]] || { echo "ERROR: expected 200 W on both cards"; exit 1; }
cool() { for _ in $(seq 1 30); do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
    [[ "$t" -le 55 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20; done; }
fin() { git add -A results logs scripts/run-h84-kvonly.sh
    git commit -q -m "bench: H84 KV-only catch-up ($1)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }; }
steps() { awk -F, -v l="$1" '$1==l && $16=="ok"{d+=$6; n++; s+=($8-$10); t+=$8/$6; a+=$11} END{if(n) printf "%s: decode %.2f, accept %.1f%%, %.2f ms/target-step (n=%d)\n", l, d/n, a/n, 1000*t/s, n}' results/h11-placement.csv; }

export N_PREDICT=400 BATCH=2048 UB=2048 BIN=$FORK/llama-server REPS=3 SPLIT=tensor
cool; CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt ./scripts/run-spec-placement.sh h84-kvonly-16k draft-mtp "$MTP" default
steps h83-splitstate-16k; steps h84-kvonly-16k; cmp=$(python3 scripts/compare-outputs.py h83-splitstate-16k h84-kvonly-16k); echo "$cmp"; fin 16k
if ! echo "$cmp" | grep -q identical; then
  cool; KV=f16 SPEC=mtp ./scripts/run-phase8-tier2.sh h84-kvonly-f16
  hs=$(awk -F, '$1=="h84-kvonly-f16"{c+=$5} END{print c+0}' results/phase8-tier2.csv); echo "haystack: $hs/19 (h58 19/19)"; fin haystack
else echo "output identical: haystack skipped"; fi
echo "=== H84 done ==="
