#!/usr/bin/env bash
# Final-report resume (user, 2026-10-16), after the r10 run was stopped for noise.
# 1) Stock 100k at 200 W, REPS=3 (user call): same stock config as r10s-stock-16k (rebased build, P2P, MTP Q4_0)
#    at the banked stock 100k settings (h26-100k-mtp-kvq8: q8_0 KV, -c 100000, np 1). Only stock 100k data is 175 W.
# 2) 100k haystack, current fork serve command with f16 KV (Phase 8 tier 2). Pass = every question p8-ref answers.
# The 1-card r10 arm is deferred (user call).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK=/root/llama.cpp-Pascal/build-cuda-p100/bin/llama-server
SERVE_ENV="GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 GGML_CUDA_PASCAL_GEMM_ALGO=1 GGML_PASCAL_DUPLEX_REDUCE=1 GGML_PASCAL_MMVH_Q6K_WIDE=1 GGML_PASCAL_MMVH_LOWBIT=1 GGML_PASCAL_FA_GEMM=1"
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 SPLIT=tensor N_PREDICT=400 BATCH=2048 UB=2048
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader | paste -sd' ')
echo "cap: $cap"; [[ "$cap" == "200.00 W 200.00 W" ]] || { echo "ERROR: expected 200 W on both cards"; exit 1; }
cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 60 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20
    done
}
commit() { git add -A results logs scripts/run-r10s-resume.sh
    git commit -q -m "bench: final report ($1)" && { git push -q && echo "pushed $1" || echo "WARNING: push failed" >&2; }; }

cool
( REPS=3 CTX=100000 CTK=q8_0 CTV=q8_0 NP=1 REQ_TIMEOUT=1800 PROMPT_FILE=prompts/h26-100k-prompt.txt \
  ./scripts/run-spec-placement.sh r10s-stock-100k draft-mtp /root/mtp-Qwen3.8-27B-Q4_0.gguf default ); echo "--- r10s-stock-100k exit $? ---"
commit r10s-stock-100k
cool
( export $SERVE_ENV; BIN=$FORK KV=f16 SPEC=mtp ./scripts/run-phase8-tier2.sh p8-final-f16 ); echo "--- p8-final-f16 exit $? ---"
awk -F, '$1=="p8-ref"||$1=="p8-h46-fagemm"||$1=="p8-final-f16"{n[$1]++; h[$1]+=$5} END{for(l in n) print "haystack", l, h[l]"/"n[l]}' results/phase8-tier2.csv
awk -F, '$1=="p8-ref"&&$5==1{r[$2","$3]=1} $1=="p8-final-f16"&&$5!=1&&r[$2","$3]{print "MISS vs ref:", $2, $3}' results/phase8-tier2.csv
commit p8-final-f16
echo "=== done ==="
