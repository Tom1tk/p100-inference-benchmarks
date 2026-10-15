#!/usr/bin/env bash
# Headline re-measure at REPS=10, for the FINAL report only (user, 2026-10-15). Stopped early on first use;
# r10-stock-16k (10 reps) is banked, r10-fork-16k partial. Re-run the fork arms once the levers are done.
# Measurement only, no lever. Arms: stock baseline 16k (as d5-p2p-mtp-16k), current serve command
# 16k (as h44-lowbit-16k), current serve command 98k (as serve-fork-100k). Commit after each arm.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK=/root/llama.cpp-Pascal/build-cuda-p100/bin/llama-server
MTP=/root/mtp-Qwen3.8-27B-Q4_0.gguf
SERVE_ENV="GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 GGML_CUDA_PASCAL_GEMM_ALGO=1 GGML_PASCAL_DUPLEX_REDUCE=1 GGML_PASCAL_MMVH_Q6K_WIDE=1 GGML_PASCAL_MMVH_LOWBIT=1"
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 SPLIT=tensor REPS=10 N_PREDICT=400 BATCH=2048 UB=2048
cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 60 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20
    done
}
commit() { git add -A results logs scripts/run-headline-r10.sh
    git commit -q -m "bench: headline REPS=10 ($1)" && { git push -q && echo "pushed $1" || echo "WARNING: push failed" >&2; }; }

cool
( CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt \
  ./scripts/run-spec-placement.sh r10-stock-16k draft-mtp $MTP default ); echo "--- r10-stock-16k exit $? ---"; commit r10-stock-16k
cool
( export $SERVE_ENV; CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt BIN=$FORK \
  ./scripts/run-spec-placement.sh r10-fork-16k draft-mtp $MTP default ); echo "--- r10-fork-16k exit $? ---"; commit r10-fork-16k
cool
( export $SERVE_ENV; CTX=100000 CTK=q8_0 CTV=q8_0 NP=1 REQ_TIMEOUT=1800 PROMPT_FILE=prompts/h26-100k-prompt.txt BIN=$FORK \
  ./scripts/run-spec-placement.sh r10-fork-100k draft-mtp $MTP default ); echo "--- r10-fork-100k exit $? ---"; commit r10-fork-100k
python3 - <<'PY'
import csv, statistics as st
rows = [r for r in csv.DictReader(open("results/h11-placement.csv")) if r["label"].startswith("r10-") and r["outcome"] == "ok"]
for lab in dict.fromkeys(r["label"] for r in rows):
    d = [float(r["decode_tps"]) for r in rows if r["label"] == lab]
    p = [float(r["prefill_tps"]) for r in rows if r["label"] == lab]
    print(f"{lab}: n={len(d)} decode {st.mean(d):.2f} ±{st.stdev(d) if len(d)>1 else 0:.2f}  prefill {st.mean(p):.1f} ±{st.stdev(p) if len(p)>1 else 0:.1f}")
PY
echo "=== done ==="
