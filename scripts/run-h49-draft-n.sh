#!/usr/bin/env bash
# H49: MTP draft length (--spec-draft-n-max, default 3) at the 200 W running cap. Decode-only lever.
# 16k (same settings as the control p15-200w-16k: default f16 KV, REPS=3):
#   control p15-200w-16k (2 reps) + p15-200w-16k-r3 (1 rep); n=4 then n=5 only if n=4 beat the control; n=2.
# 98k (CACHE_PROMPT=1: rep 1 prefills, reps 2-3 reuse the prompt KV; decode is the metric), KV98 from H48:
#   n=3 control under the same cache path, n=2, and the best up-step from 16k only if it won there.
# Pass: decode >= 2% over the control mean at either depth, with identical output (greedy verification).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 GGML_CUDA_PASCAL_GEMM_ALGO=1 \
       GGML_PASCAL_DUPLEX_REDUCE=1 GGML_PASCAL_MMVH_Q6K_WIDE=1 GGML_PASCAL_MMVH_LOWBIT=1 GGML_PASCAL_FA_GEMM=1
export N_PREDICT=400 BATCH=2048 UB=2048 BIN=/root/llama.cpp-Pascal/build-cuda-p100/bin/llama-server
KV98="${KV98:?set KV98 to the KV type H48 settled (q8_0 or f16)}"
MTP=/root/mtp-Qwen3.8-27B-Q4_0-head96k.gguf
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader | paste -sd' ')
echo "cap: $cap"; [[ "$cap" == "200.00 W 200.00 W" ]] || { echo "ERROR: expected 200 W on both cards"; exit 1; }
cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 55 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20
    done
}
fin() { git add -A results logs scripts/run-h49-draft-n.sh
    git commit -q -m "bench: H49 draft length ($1)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }; }
mean() { awk -F, -v re="$1" '$1 ~ re && $16=="ok" {s+=$6; n++} END {if (n) printf "%.2f", s/n; else print 0}' results/h11-placement.csv; }
gt() { python3 -c "import sys; sys.exit(0 if float('$1') > float('$2') else 1)"; }
run16k() {  # label n reps
    cool
    EXTRA_ARGS="--spec-draft-n-max $2" REPS=$3 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt SPLIT=tensor \
        ./scripts/run-spec-placement.sh "$1" draft-mtp "$MTP" default
    python3 scripts/compare-outputs.py p15-200w-16k "$1" || echo "WARNING: $1 output differs from control"
    fin "$1"
}
run98k() {  # label n
    cool
    EXTRA_ARGS="--spec-draft-n-max $2" CACHE_PROMPT=1 REPS=3 CTX=100000 CTK=$KV98 CTV=$KV98 PROMPT_FILE=prompts/h26-100k-prompt.txt \
        NP=1 REQ_TIMEOUT=1800 SPLIT=tensor ./scripts/run-spec-placement.sh "$1" draft-mtp "$MTP" default
    [[ "$1" != h49-n3-98k ]] && { python3 scripts/compare-outputs.py h49-n3-98k "$1" || echo "WARNING: $1 output differs from control"; }
    fin "$1"
}

# --- 16k ---
run16k p15-200w-16k-r3 3 1
C16=$(mean '^p15-200w-16k(-r3)?$'); echo "16k control decode mean $C16"
run16k h49-n4-16k 4 3; N4=$(mean '^h49-n4-16k$'); echo "16k n=4 $N4 vs control $C16"
UP=""; UPV=$C16
if gt "$N4" "$C16"; then
    UP=4; UPV=$N4
    run16k h49-n5-16k 5 3; N5=$(mean '^h49-n5-16k$'); echo "16k n=5 $N5 vs n=4 $N4"
    gt "$N5" "$N4" && { UP=5; UPV=$N5; }
else
    echo "skip: n=4 did not beat the control, n=5 cancelled"
fi
run16k h49-n2-16k 2 3; N2=$(mean '^h49-n2-16k$'); echo "16k n=2 $N2 vs control $C16"

# --- 98k ---
run98k h49-n3-98k 3
run98k h49-n2-98k 2
if [[ -n "$UP" ]]; then run98k "h49-n${UP}-98k" "$UP"; else echo "skip: no up-step won at 16k, 98k up arm cancelled"; fi

echo "=== decode means ==="
for l in p15-200w-16k h49-n2-16k h49-n4-16k h49-n5-16k h49-n3-98k h49-n2-98k h49-n4-98k h49-n5-98k; do
    printf '%-16s %s\n' "$l" "$(mean "^${l}(-r3)?\$")"
done
grep -E '^(label|p15-200w-16k|h49-)' results/h11-placement.csv | cut -d, -f1,5-8,16 | column -s, -t
cat logs/h49-*.cap 2>/dev/null
fin "summary"
