#!/usr/bin/env bash
# H76: conv state update copies in one launch (GGML_PASCAL_CONV_STATE_CPY, default on sm_60): CONCAT -> CPY
# fused, and the K = n_rs_seq + 1 rollback snapshot CPYs batched. llama-bench skipped (n_rs_seq = 0 there:
# only the concat half, ~0.5% predicted, under its noise). 1) 16k serve REPS=3 vs banked h75-on-16k (the
# lever-off path), both fire logs required, output must be byte-identical. 2) haystack only if it is not.
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
fin() { git add -A results logs scripts/run-h76-convcpy.sh
    git commit -q -m "bench: H76 conv state copies ($1)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }; }
steps() { awk -F, -v l="$1" '$1==l && $16=="ok"{d+=$6; n++; s+=($8-$10); t+=$8/$6; a+=$11} END{if(n) printf "%s: decode %.2f, accept %.1f%%, %.2f ms/target-step (n=%d)\n", l, d/n, a/n, 1000*t/s, n}' results/h11-placement.csv; }

export N_PREDICT=400 BATCH=2048 UB=2048 BIN=$FORK/llama-server REPS=3 SPLIT=tensor
cool; CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt ./scripts/run-spec-placement.sh h76-on-16k draft-mtp "$MTP" default
grep -h "conv state" logs/h76-on-16k.server.log | sort -u
grep -q "cpy batch fired" logs/h76-on-16k.server.log && grep -q "concat+cpy fired" logs/h76-on-16k.server.log || echo "ERROR: lever did not fire"
steps h75-on-16k; steps h76-on-16k; cmp=$(python3 scripts/compare-outputs.py h75-on-16k h76-on-16k); echo "$cmp"; fin 16k
if ! echo "$cmp" | grep -q identical; then
  cool; KV=f16 SPEC=mtp ./scripts/run-phase8-tier2.sh h76-on-f16
  hs=$(awk -F, '$1=="h76-on-f16"{c+=$5} END{print c+0}' results/phase8-tier2.csv); echo "haystack: $hs/19 (h58 19/19)"; fin haystack
else echo "output identical: haystack skipped"; fi
echo "=== H76 done ==="
