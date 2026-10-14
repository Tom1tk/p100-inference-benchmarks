#!/usr/bin/env bash
# H35: does GEMM ALGO3 add quality *loss*, or only a different FP16 rounding?
# H34 gated ALGO3 against the default FP16 GEMM (difference). Here ALGO3 is measured against
# H30's exact reference (Q5_K_M, sm_60 FP32, build-h17) and compared with the banked
# q4-fp16 arm (same config, default GEMM): mean 0.062355, same-top 95.833.
# If it passes, the two speed arms that H34 skipped run with ALGO3 on.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FORK=/root/llama.cpp-Pascal/build-cuda-p100/bin
FP32=/root/llama.cpp-Pascal/build-h17/bin/llama-perplexity
Q5=/root/Qwen3.8-27B-UD-Q5_K_M.gguf
Q4=/root/Qwen3.8-27B-UD-Q4_K_M.gguf
IQ3=/root/Qwen3.8-27B-UD-IQ3_S.gguf
KLDIR=/root/h28-kld; mkdir -p "$KLDIR" logs
BASE=$KLDIR/h35-base.kld
CSV=results/h35-kld.csv
export GGML_CUDA_ALLREDUCE=none

cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 60 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20
    done
}
ppl() {  # label bin model algo extra-args...  (H30 config: 2-card tensor, P2P, ub 512)
    local label=$1 bin=$2 model=$3 algo=$4; shift 4
    cool
    ABORT_TEMP=80 INTERVAL=2 ./scripts/gpu-monitor.sh > "logs/h35-$label-temps.log" 2>/dev/null &
    local mon=$!
    GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_GEMM_ALGO=$algo "$bin" -m "$model" -f prompts/h26-100k-prompt.txt \
        -c 512 --chunks 8 -b 512 -ub 512 -ngl 99 -fa 1 -sm tensor "$@" > "logs/h35-$label.log" 2>&1
    local rc=$?
    kill $mon
    echo "--- $label rc=$rc, peak $(tr '|' '\n' < "logs/h35-$label-temps.log" | awk -F', ' 'NF>=3{print $2}' | sort -n | tail -1)C ---"
    return $rc
}

echo "label,mean_kld,p99_kld,p999_kld,same_top_pct" > "$CSV"
ppl ref-q5-fp32 "$FP32" "$Q5" 0 --kl-divergence-base "$BASE" || exit 1
ppl q4-fp16-algo3 "$FORK/llama-perplexity" "$Q4" 1 --kl-divergence-base "$BASE" --kl-divergence || exit 1
f=logs/h35-q4-fp16-algo3.log
printf '%s,%s,%s,%s,%s\n' q4-fp16-algo3 "$(grep -m1 'Mean    KLD:' $f | awk '{print $3}')" \
    "$(grep -m1 '99.0%   KLD:' $f | awk '{print $3}')" "$(grep -m1 '99.9%   KLD:' $f | awk '{print $3}')" \
    "$(grep -m1 'Same top p:' $f | awk '{print $4}')" >> "$CSV"
rm -f "$BASE"
column -s, -t "$CSV"

# gate: no more than 5% above the default-GEMM arm's loss, top-1 within 0.5 points of it
PASS=$(python3 - "$CSV" <<'PY'
import csv, sys
r = next(x for x in csv.DictReader(open(sys.argv[1])) if x["label"] == "q4-fp16-algo3")
print(int(float(r["mean_kld"]) <= 0.062355 * 1.05 and float(r["same_top_pct"]) >= 95.833 - 0.5))
PY
)
echo "gate: algo3=$PASS"

if [[ $PASS == 1 ]]; then
    # Speed arms: H34's, plus ALGO3 (one lever). Controls (rep 2): h34-2card-f16reduce
    # 26.64 / 360.71, h34-1card-iq3 15.83 / 196.72.
    export REPS=2 N_PREDICT=400 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt BATCH=2048 UB=2048 \
           BIN=$FORK/llama-server GGML_CUDA_PASCAL_MMVH=1 GGML_CUDA_PASCAL_GEMM_ALGO=1
    arm() { cool; "$@"; echo "--- done (exit $?) ---"; sleep 15; }
    arm env GGML_CUDA_P2P=0 GGML_PASCAL_F16_REDUCE=1 SPLIT=tensor MODEL="$Q4" \
        ./scripts/run-spec-placement.sh h35-2card-algo3 none none default
    arm env SPLIT=none MODEL="$IQ3" ./scripts/run-spec-placement.sh h35-1card-iq3-algo3 none none default
    grep -E '^(label|h34-2card-f16reduce|h34-1card-iq3|h35-)' results/h11-placement.csv | column -s, -t
fi

git add -A results logs scripts/run-h35-algo3-loss.sh
git commit -q -m "bench: H35 GEMM ALGO3 loss vs the FP32 Q5 reference (+ speed arms if it passes)" \
    && { git push -q && echo pushed || echo "WARNING: push failed; commit is local" >&2; }
