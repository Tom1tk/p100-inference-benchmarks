#!/usr/bin/env bash
# H30 (Phase 8 tier 1): decompose quality loss into quant, arithmetic (H17) and kernel (mmvh)
# by KLD against one reference: Q5_K_M with exact sm_60 FP32 arithmetic (fork build-h17).
# Every arm is 8x512 tokens of prompts/h26-100k-prompt.txt, as in H28/H29.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FORK=/root/llama.cpp-Pascal/build-cuda-p100/bin/llama-perplexity
FP32=/root/llama.cpp-Pascal/build-h17/bin/llama-perplexity
Q5=/root/Qwen3.8-27B-UD-Q5_K_M.gguf
Q4=/root/Qwen3.8-27B-UD-Q4_K_M.gguf
IQ3=/root/Qwen3.8-27B-UD-IQ3_S.gguf
KLDIR=/root/h28-kld; mkdir -p "$KLDIR" logs
BASE=$KLDIR/h30-base.kld
CSV=results/h30-kld.csv
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1

cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 60 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20
    done
}
ppl() {  # label bin model mmvh ub split extra-args...
    local label=$1 bin=$2 model=$3 mmvh=$4 ub=$5 split=$6; shift 6
    cool
    ABORT_TEMP=80 INTERVAL=2 ./scripts/gpu-monitor.sh > "logs/h30-$label-temps.log" 2>/dev/null &
    local mon=$!
    GGML_CUDA_PASCAL_MMVH=$mmvh "$bin" -m "$model" -f prompts/h26-100k-prompt.txt \
        -c 512 --chunks 8 -b 512 -ub "$ub" -ngl 99 -fa 1 -sm "$split" "$@" > "logs/h30-$label.log" 2>&1
    local rc=$?
    kill $mon
    echo "--- $label rc=$rc, peak $(tr '|' '\n' < "logs/h30-$label-temps.log" | awk -F', ' 'NF>=3{print $2}' | sort -n | tail -1)C ---"
    return $rc
}
arm() {  # label bin model mmvh ub split
    if ppl "$@" --kl-divergence-base "$BASE" --kl-divergence; then
        local f="logs/h30-$1.log"
        printf '%s,%s,%s,%s,%s\n' "$1" \
            "$(grep -m1 'Mean    KLD:' "$f" | awk '{print $3}')" \
            "$(grep -m1 '99.0%   KLD:' "$f" | awk '{print $3}')" \
            "$(grep -m1 '99.9%   KLD:' "$f" | awk '{print $3}')" \
            "$(grep -m1 'Same top p:' "$f" | awk '{print $4}')" >> "$CSV"
    else
        echo "$1,FAILED,,," >> "$CSV"
    fi
}

echo "label,mean_kld,p99_kld,p999_kld,same_top_pct" > "$CSV"
ppl ref-q5-fp32 "$FP32" "$Q5" 0 512 tensor --kl-divergence-base "$BASE" || exit 1
arm q5-fp16        "$FORK" "$Q5"  0 512 tensor   # H17 alone
arm q4-fp32        "$FP32" "$Q4"  0 512 tensor   # quant alone
arm q4-fp16        "$FORK" "$Q4"  0 512 tensor   # serve prefill numerics
arm q4-fp16-ub8    "$FORK" "$Q4"  0 8   tensor   # serve decode numerics, mmvq
arm q4-fp16-mmvh   "$FORK" "$Q4"  1 8   tensor   # candidate decode numerics
arm iq3-1card-ub8  "$FORK" "$IQ3" 0 8   none     # 1-card decode, mmvq
arm iq3-1card-mmvh "$FORK" "$IQ3" 1 8   none     # 1-card candidate
rm -f "$BASE"
column -s, -t "$CSV"

git add -A results logs scripts/run-h30-kld-ladder.sh
git commit -q -m "bench: H30 Phase 8 tier 1 — KLD ladder vs Q5_K_M FP32 reference" \
    && { git push -q && echo pushed || echo "WARNING: push failed; commit is local" >&2; }
