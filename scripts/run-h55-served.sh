#!/usr/bin/env bash
# H55 served: GGML_PASCAL_MMVH_R4=1 (fork: 4 rows per warp at 5-8 mmvh columns) with a longer MTP draft.
# 16k n=4 REPS=3 vs banked n=3 control r10s-fork-16k (same build bar this change, which leaves <=4 columns untouched).
# Pass: >= 3% decode over the control, output identical. If n=4 passes: 16k n=5, then 98k at the best n
# (CACHE_PROMPT=1) vs banked h49-n3-98k. Each failed step stops the driver.
# No 1-card arm: 1 card runs drafter-free (MTP OOMs), so mmvh only sees 1 column there and the path never runs.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 GGML_CUDA_PASCAL_GEMM_ALGO=1 \
       GGML_PASCAL_DUPLEX_REDUCE=1 GGML_PASCAL_MMVH_Q6K_WIDE=1 GGML_PASCAL_MMVH_LOWBIT=1 GGML_PASCAL_FA_GEMM=1 GGML_PASCAL_MMVH_R4=1
export N_PREDICT=400 BATCH=2048 UB=2048 BIN=/root/llama.cpp-Pascal/build-cuda-p100/bin/llama-server
MTP=/root/mtp-Qwen3.8-27B-Q4_0-head96k.gguf
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader | paste -sd' ')
echo "cap: $cap"; [[ "$cap" == "200.00 W 200.00 W" ]] || { echo "ERROR: expected 200 W on both cards"; exit 1; }
cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 55 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20
    done
}
fin() { git add -A results logs scripts/run-h55-served.sh
    git commit -q -m "bench: H55 MMVH_R4 served ($1)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }; }
mean() { awk -F, -v re="$1" '$1 ~ re && $16=="ok" {s+=$6; n++} END {if (n) printf "%.2f", s/n; else print 0}' results/h11-placement.csv; }
ge() { python3 -c "import sys; sys.exit(0 if float('$1') >= $3*float('$2') else 1)"; }
run16k() {  # label n
    cool
    EXTRA_ARGS="--spec-draft-n-max $2" REPS=3 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt SPLIT=tensor \
        ./scripts/run-spec-placement.sh "$1" draft-mtp "$MTP" default
    python3 scripts/compare-outputs.py p15-200w-16k "$1" || echo "WARNING: $1 output differs from control"
    fin "$1"
}

C16=$(mean '^r10s-fork-16k$'); echo "16k n=3 control r10s-fork-16k: $C16"
run16k h55-r4-n4-16k 4; N4=$(mean '^h55-r4-n4-16k$'); echo "16k n=4 R4: $N4 vs control $C16 (H49 n=4 without R4: 35.08)"
ge "$N4" "$C16" 1.03 || { echo "skip: n=4 R4 under +3%, n=5 and 98k cancelled"; exit 0; }
BEST=4; BV=$N4
run16k h55-r4-n5-16k 5; N5=$(mean '^h55-r4-n5-16k$'); echo "16k n=5 R4: $N5 vs n=4 $N4"
ge "$N5" "$N4" 1.0 && { BEST=5; BV=$N5; }
cool
EXTRA_ARGS="--spec-draft-n-max $BEST" CACHE_PROMPT=1 REPS=3 CTX=100000 CTK=f16 CTV=f16 PROMPT_FILE=prompts/h26-100k-prompt.txt \
    NP=1 REQ_TIMEOUT=1800 SPLIT=tensor ./scripts/run-spec-placement.sh "h55-r4-n${BEST}-98k" draft-mtp "$MTP" default
python3 scripts/compare-outputs.py h49-n3-98k "h55-r4-n${BEST}-98k" || echo "WARNING: 98k output differs from control"
echo "98k n=$BEST R4: $(mean "^h55-r4-n${BEST}-98k\$") vs n=3 control h49-n3-98k $(mean '^h49-n3-98k$')"
grep -E '^(r10s-fork-16k|h49-n3-98k|h55-)' results/h11-placement.csv | cut -d, -f1,5-8,11,16 | column -s, -t
cat logs/h55-*.cap 2>/dev/null
fin "summary"
