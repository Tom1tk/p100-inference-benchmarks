#!/usr/bin/env bash
# H48: f16 KV instead of q8_0 at 98k, serve command at the 200 W running cap.
# Control: p15-200w-98k (2 reps) + p15-200w-98k-r3 (1 rep) = 3.
# Arm h48-f16kv-98k rep 1; reps 2-3 (-r23) only if prefill > 1.015x control mean.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 GGML_CUDA_PASCAL_GEMM_ALGO=1 \
       GGML_PASCAL_DUPLEX_REDUCE=1 GGML_PASCAL_MMVH_Q6K_WIDE=1 GGML_PASCAL_MMVH_LOWBIT=1 GGML_PASCAL_FA_GEMM=1
export N_PREDICT=400 BATCH=2048 UB=2048 BIN=/root/llama.cpp-Pascal/build-cuda-p100/bin/llama-server
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader | paste -sd' ')
echo "cap: $cap"; [[ "$cap" == "200.00 W 200.00 W" ]] || { echo "ERROR: expected 200 W on both cards"; exit 1; }
cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 55 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20
    done
}
run98k() {  # label kv reps
    cool
    REPS=$3 CTX=100000 CTK=$2 CTV=$2 PROMPT_FILE=prompts/h26-100k-prompt.txt NP=1 REQ_TIMEOUT=1800 SPLIT=tensor \
        ./scripts/run-spec-placement.sh "$1" draft-mtp /root/mtp-Qwen3.8-27B-Q4_0-head96k.gguf default
}
fin() { git add -A results logs scripts/run-h48-f16kv-98k.sh scripts/run-spec-placement.sh
    git commit -q -m "bench: H48 f16 KV at 98k ($1)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }; }
mean() { awk -F, -v re="$1" -v c="$2" '$1 ~ re && $16=="ok" {s+=$c; n++} END {if (n) printf "%.2f", s/n; else print 0}' results/h11-placement.csv; }

run98k p15-200w-98k-r3 q8_0 1
fin "control rep 3"
CTRL=$(mean '^p15-200w-98k(-r3)?$' 7)
run98k h48-f16kv-98k f16 1 || { fin "f16 arm failed (OOM?)"; exit 1; }
G1=$(mean '^h48-f16kv-98k$' 7)
echo "98k prefill: control mean $CTRL (3 reps), f16 rep 1 $G1"
if python3 -c "import sys; sys.exit(0 if float('$G1') > 1.015*float('$CTRL') else 1)"; then
    run98k h48-f16kv-98k-r23 f16 2
else
    echo "skip: f16 rep 1 not 1.5% over control, reps 2-3 cancelled"
fi
python3 scripts/compare-outputs.py p15-200w-98k h48-f16kv-98k || true
grep -E '^(label|p15-200w-98k|h48-)' results/h11-placement.csv | column -s, -t
cat logs/h48-f16kv-98k*.cap logs/p15-200w-98k-r3.cap 2>/dev/null
fin "arm"
