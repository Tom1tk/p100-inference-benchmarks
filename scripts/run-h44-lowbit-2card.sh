#!/usr/bin/env bash
# H44: GGML_PASCAL_MMVH_LOWBIT=1 (Q3_K + IQ4_NL on mmvh, fork 2d3b3bf) on the 2-card serve command.
# 1) KLD vs the Q5_K_M FP32 reference (build-h17), Q4_K_M -ub 8 -sm tensor, serve env, lowbit off vs on.
#    Pass: on mean <= off mean * 1.05, same top >= off - 0.5. Fail -> no speed arm.
# 2) 2-card 16k speed, REPS=2. Control banked: h43-k4scale-16k 42.39 / 366.4.
# 1-card arm parked by rule zero (HYPOTHESES H44).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK=/root/llama.cpp-Pascal/build-cuda-p100/bin
FP32=/root/llama.cpp-Pascal/build-h17/bin/llama-perplexity
Q5=/root/Qwen3.8-27B-UD-Q5_K_M.gguf
Q4=/root/Qwen3.8-27B-UD-Q4_K_M.gguf
BASE=/root/h28-kld/h44-base.kld; mkdir -p /root/h28-kld logs
CSV=results/h44-kld.csv
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1
SERVE_ENV="GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 GGML_CUDA_PASCAL_GEMM_ALGO=1 GGML_PASCAL_DUPLEX_REDUCE=1 GGML_PASCAL_MMVH_Q6K_WIDE=1"
cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 60 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20
    done
}
ppl() {  # label bin model extra-args...
    local label=$1 bin=$2 model=$3; shift 3
    cool
    ABORT_TEMP=80 INTERVAL=2 ./scripts/gpu-monitor.sh > "logs/h44-$label-temps.log" 2>/dev/null &
    local mon=$!
    "$bin" -m "$model" -f prompts/h26-100k-prompt.txt -c 2048 --chunks 2 -b 2048 -ngl 99 -fa 1 "$@" > "logs/h44-$label.log" 2>&1
    local rc=$?; kill $mon
    echo "--- $label rc=$rc, peak $(tr '|' '\n' < "logs/h44-$label-temps.log" | awk -F', ' 'NF>=3{print $2}' | sort -n | tail -1)C, $(grep -m1 -E 'Mean    KLD' "logs/h44-$label.log") ---"
    return $rc
}
row() { local f=logs/h44-$1.log
    printf '%s,%s,%s,%s,%s\n' "$1" "$(grep -m1 'Mean    KLD:' $f | awk '{print $3}')" \
        "$(grep -m1 '99.0%   KLD:' $f | awk '{print $3}')" "$(grep -m1 '99.9%   KLD:' $f | awk '{print $3}')" \
        "$(grep -m1 'Same top p:' $f | awk '{print $4}')" >> "$CSV"; }
fin() { rm -f "$BASE"; git add -A results logs scripts/run-h44-lowbit-2card.sh HYPOTHESES.md
    git commit -q -m "bench: H44 low-bit mmvh 2-card ($1)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }; }

echo "label,mean_kld,p99_kld,p999_kld,same_top_pct" > "$CSV"
ppl ref-q5-fp32 "$FP32" "$Q5" -ub 512 -sm tensor --kl-divergence-base "$BASE" || { fin "reference failed"; exit 1; }
for lb in off on; do
    ( export $SERVE_ENV GGML_PASCAL_MMVH_LOWBIT=$([[ $lb == on ]] && echo 1 || echo 0)
      ppl q4-lowbit-$lb "$FORK/llama-perplexity" "$Q4" -ub 8 -sm tensor --kl-divergence-base "$BASE" --kl-divergence ) \
        || { fin "perplexity failed"; exit 1; }
    row q4-lowbit-$lb
done
column -s, -t "$CSV"
python3 - "$CSV" <<'PY' || { echo "gate: KLD FAIL"; fin "KLD FAILED"; exit 1; }
import csv, sys
r = {x["label"]: x for x in csv.DictReader(open(sys.argv[1]))}
off, on = r["q4-lowbit-off"], r["q4-lowbit-on"]
sys.exit(0 if float(on["mean_kld"]) <= float(off["mean_kld"]) * 1.05
         and float(on["same_top_pct"]) >= float(off["same_top_pct"]) - 0.5 else 1)
PY
echo "gate: KLD PASS"

export $SERVE_ENV GGML_PASCAL_MMVH_LOWBIT=1
export N_PREDICT=400 BATCH=2048 UB=2048 BIN=$FORK/llama-server
cool
REPS=2 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt SPLIT=tensor \
    ./scripts/run-spec-placement.sh h44-lowbit-16k draft-mtp /root/mtp-Qwen3.8-27B-Q4_0.gguf default
python3 scripts/compare-outputs.py h43-k4scale-16k h44-lowbit-16k || true
grep -E '^(label|h43-k4scale-16k|h44-)' results/h11-placement.csv | column -s, -t
fin "KLD + 2-card 16k"
