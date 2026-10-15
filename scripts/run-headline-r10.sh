#!/usr/bin/env bash
# Headline re-measure at REPS=10, for the FINAL report only (user, 2026-10-15). First use (r10-*) was stopped
# early and is at 175 W; this run (r10s-*) re-takes every arm at the 200 W running cap, f16 KV everywhere.
# Measurement only, no lever. Arms: stock baseline 16k (as d5-p2p-mtp-16k), current serve command
# 16k (as h44-lowbit-16k), current serve command 98k (as serve-fork-100k), 1-card IQ3_S 16k fork arm (+DEQUANT_STAGED, H39r),
# then the 100k haystack with f16 KV (Phase 8 tier 2, pass = every question p8-ref answers). Commit after each arm.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK=/root/llama.cpp-Pascal/build-cuda-p100/bin/llama-server
MTP=/root/mtp-Qwen3.8-27B-Q4_0.gguf
MTP_FORK=/root/mtp-Qwen3.8-27B-Q4_0-head96k.gguf  # H45; stock builds reject the trimmed head
IQ3=/root/Qwen3.8-27B-UD-IQ3_S.gguf
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader | paste -sd' ')
echo "cap: $cap"; [[ "$cap" == "200.00 W 200.00 W" ]] || { echo "ERROR: expected 200 W on both cards"; exit 1; }
SERVE_ENV="GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 GGML_CUDA_PASCAL_GEMM_ALGO=1 GGML_PASCAL_DUPLEX_REDUCE=1 GGML_PASCAL_MMVH_Q6K_WIDE=1 GGML_PASCAL_MMVH_LOWBIT=1 GGML_PASCAL_FA_GEMM=1"
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 SPLIT=tensor REPS=10 N_PREDICT=400 BATCH=2048 UB=2048
cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 60 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20
    done
}
commit() { git add -A results logs scripts/run-headline-r10.sh scripts/run-phase8-tier2.sh
    git commit -q -m "bench: headline REPS=10 ($1)" && { git push -q && echo "pushed $1" || echo "WARNING: push failed" >&2; }; }

cool
( CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt \
  ./scripts/run-spec-placement.sh r10s-stock-16k draft-mtp $MTP default ); echo "--- r10s-stock-16k exit $? ---"; commit r10s-stock-16k
cool
( export $SERVE_ENV; CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt BIN=$FORK \
  ./scripts/run-spec-placement.sh r10s-fork-16k draft-mtp $MTP_FORK default ); echo "--- r10s-fork-16k exit $? ---"; commit r10s-fork-16k
cool
( export $SERVE_ENV; CTX=100000 CTK=f16 CTV=f16 NP=1 REQ_TIMEOUT=1800 PROMPT_FILE=prompts/h26-100k-prompt.txt BIN=$FORK \
  ./scripts/run-spec-placement.sh r10s-fork-100k draft-mtp $MTP_FORK default ); echo "--- r10s-fork-100k exit $? ---"; commit r10s-fork-100k
cool
( export $SERVE_ENV GGML_PASCAL_DEQUANT_STAGED=1; CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt BIN=$FORK SPLIT=none MODEL=$IQ3 \
  ./scripts/run-spec-placement.sh r10s-fork-1card-16k none none default ); echo "--- r10s-fork-1card-16k exit $? ---"; commit r10s-fork-1card-16k
cool
( export $SERVE_ENV; BIN=$FORK KV=f16 SPEC=mtp ./scripts/run-phase8-tier2.sh p8-final-f16 ); echo "--- p8-final-f16 exit $? ---"
awk -F, '$1=="p8-ref"||$1=="p8-final-f16"{n[$1]++; h[$1]+=$5} END{for(l in n) print "haystack", l, h[l]"/"n[l]}' results/phase8-tier2.csv
awk -F, '$1=="p8-ref"&&$5==1{r[$2","$3]=1} $1=="p8-final-f16"&&$5!=1&&r[$2","$3]{print "MISS vs ref:", $2, $3}' results/phase8-tier2.csv
commit p8-final-f16
python3 - <<'PY'
import csv, statistics as st
rows = [r for r in csv.DictReader(open("results/h11-placement.csv")) if r["label"].startswith("r10s-") and r["outcome"] == "ok"]
for lab in dict.fromkeys(r["label"] for r in rows):
    d = [float(r["decode_tps"]) for r in rows if r["label"] == lab]
    p = [float(r["prefill_tps"]) for r in rows if r["label"] == lab]
    print(f"{lab}: n={len(d)} decode {st.mean(d):.2f} ±{st.stdev(d) if len(d)>1 else 0:.2f}  prefill {st.mean(p):.1f} ±{st.stdev(p) if len(p)>1 else 0:.1f}")
PY
echo "=== done ==="
