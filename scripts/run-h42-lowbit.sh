#!/usr/bin/env bash
# H42: GGML_PASCAL_MMVH_LOWBIT=1 (Q3_K, Q2_K, IQ2_S/XS/XXS on mmvh), 1-card UD-IQ3_S.
# 1) KLD vs the Q5_K_M FP32 reference (build-h17), IQ3_S -ub 8 -sm none, lowbit off vs on.
#    Pass: on mean <= off mean * 1.05, same top >= off - 0.5.
# 2) 1-card 16k speed, REPS=2. Control banked: h35-1card-iq3-algo3 15.84 / 198.5.
# 2-card arm parked by rule zero (HYPOTHESES H42).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK=/root/llama.cpp-Pascal/build-cuda-p100/bin
FP32=/root/llama.cpp-Pascal/build-h17/bin/llama-perplexity
Q5=/root/Qwen3.8-27B-UD-Q5_K_M.gguf
IQ3=/root/Qwen3.8-27B-UD-IQ3_S.gguf
BASE=/root/h28-kld/h42-base.kld; mkdir -p /root/h28-kld logs
CSV=results/h42-kld.csv
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_PASCAL_GEMM_ALGO=1
cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 60 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20
    done
}
ppl() {  # label bin model extra-args...
    local label=$1 bin=$2 model=$3; shift 3
    cool
    ABORT_TEMP=80 INTERVAL=2 ./scripts/gpu-monitor.sh > "logs/h42-$label-temps.log" 2>/dev/null &
    local mon=$!
    "$bin" -m "$model" -f prompts/h26-100k-prompt.txt -c 2048 --chunks 2 -b 2048 -ngl 99 -fa 1 "$@" > "logs/h42-$label.log" 2>&1
    local rc=$?; kill $mon
    echo "--- $label rc=$rc, peak $(tr '|' '\n' < "logs/h42-$label-temps.log" | awk -F', ' 'NF>=3{print $2}' | sort -n | tail -1)C, $(grep -m1 -E 'Mean    KLD' "logs/h42-$label.log") ---"
    return $rc
}
row() { local f=logs/h42-$1.log
    printf '%s,%s,%s,%s,%s\n' "$1" "$(grep -m1 'Mean    KLD:' $f | awk '{print $3}')" \
        "$(grep -m1 '99.0%   KLD:' $f | awk '{print $3}')" "$(grep -m1 '99.9%   KLD:' $f | awk '{print $3}')" \
        "$(grep -m1 'Same top p:' $f | awk '{print $4}')" >> "$CSV"; }
fin() { rm -f "$BASE"; git add -A results logs scripts/run-h42-lowbit.sh HYPOTHESES.md
    git commit -q -m "bench: H42 low-bit mmvh 1-card ($1)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }; }

echo "label,mean_kld,p99_kld,p999_kld,same_top_pct" > "$CSV"
GGML_CUDA_P2P=1 ppl ref-q5-fp32 "$FP32" "$Q5" -ub 512 -sm tensor --kl-divergence-base "$BASE" || { fin "reference failed"; exit 1; }
for lb in off on; do
    GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_MMVH_LOWBIT=$([[ $lb == on ]] && echo 1 || echo 0) \
        ppl iq3-lowbit-$lb "$FORK/llama-perplexity" "$IQ3" -ub 8 -sm none -mg 0 --kl-divergence-base "$BASE" --kl-divergence \
        || { fin "perplexity failed"; exit 1; }
    row iq3-lowbit-$lb
done
column -s, -t "$CSV"
python3 - "$CSV" <<'PY' || { echo "gate: KLD FAIL"; fin "KLD FAILED"; exit 1; }
import csv, sys
r = {x["label"]: x for x in csv.DictReader(open(sys.argv[1]))}
off, on = r["iq3-lowbit-off"], r["iq3-lowbit-on"]
sys.exit(0 if float(on["mean_kld"]) <= float(off["mean_kld"]) * 1.05
         and float(on["same_top_pct"]) >= float(off["same_top_pct"]) - 0.5 else 1)
PY
echo "gate: KLD PASS"

export REPS=2 N_PREDICT=400 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt BATCH=2048 UB=2048 \
       BIN=$FORK/llama-server GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_MMVH_LOWBIT=1
cool
SPLIT=none MODEL="$IQ3" ./scripts/run-spec-placement.sh h42-1card-iq3-lowbit none none default
grep -E '^(label|h35-1card-iq3-algo3|h42-)' results/h11-placement.csv | column -s, -t
fin "KLD + 1-card 16k"
