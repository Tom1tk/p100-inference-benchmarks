#!/usr/bin/env bash
# H31: fork P1, F16 AllReduce payload for prefill (GGML_PASCAL_F16_REDUCE=1), 2-card only.
# KLD gate first (ub 512, so every reduce is >= 1 MiB and goes F16), then one 16k speed arm
# against the banked h28-2card-mmvh1 control (same binary path, env off).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FORKBIN=/root/llama.cpp-Pascal/build-cuda-p100/bin
Q4=/root/Qwen3.8-27B-UD-Q4_K_M.gguf
KLDIR=/root/h28-kld; mkdir -p "$KLDIR" logs
CSV=results/h31-kld.csv
export GGML_CUDA_ALLREDUCE=none

cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 60 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20
    done
}
ppl() {  # label f16reduce extra-args...
    local label=$1 f16=$2; shift 2
    cool
    ABORT_TEMP=80 INTERVAL=2 ./scripts/gpu-monitor.sh > "logs/h31-$label-temps.log" 2>/dev/null &
    local mon=$!
    GGML_PASCAL_F16_REDUCE=$f16 "$FORKBIN/llama-perplexity" -m "$Q4" -f prompts/h26-100k-prompt.txt \
        -c 512 --chunks 8 -b 512 -ub 512 -ngl 99 -fa 1 -sm tensor "$@" > "logs/h31-$label.log" 2>&1
    local rc=$?
    kill $mon
    echo "--- $label rc=$rc, peak $(tr '|' '\n' < "logs/h31-$label-temps.log" | awk -F', ' 'NF>=3{print $2}' | sort -n | tail -1)C ---"
    return $rc
}

echo "label,mean_kld,p99_kld,p999_kld,same_top_pct" > "$CSV"
ppl base-f32 0 --kl-divergence-base "$KLDIR/h31-base.kld" || exit 1
if ppl f16-reduce 1 --kl-divergence-base "$KLDIR/h31-base.kld" --kl-divergence; then
    f=logs/h31-f16-reduce.log
    printf 'f16-reduce,%s,%s,%s,%s\n' "$(grep -m1 'Mean    KLD:' $f | awk '{print $3}')" \
        "$(grep -m1 '99.0%   KLD:' $f | awk '{print $3}')" "$(grep -m1 '99.9%   KLD:' $f | awk '{print $3}')" \
        "$(grep -m1 'Same top p:' $f | awk '{print $4}')" >> "$CSV"
fi
rm -f "$KLDIR/h31-base.kld"
column -s, -t "$CSV"

GATE=$(python3 - "$CSV" <<'PY'
import csv, sys
r = {x["label"]: x for x in csv.DictReader(open(sys.argv[1]))}
m = r.get("f16-reduce")
print("PASS" if m and float(m["mean_kld"]) <= 0.002 and float(m["same_top_pct"]) >= 99.0 else "FAIL")
PY
)
echo "KLD gate: $GATE"

if [[ "$GATE" == PASS ]]; then
    cool
    GGML_PASCAL_F16_REDUCE=1 GGML_CUDA_PASCAL_MMVH=1 REPS=2 N_PREDICT=400 CTX=16384 \
        PROMPT_FILE=prompts/h11-depth-prompt.txt BATCH=2048 UB=2048 BIN=$FORKBIN/llama-server \
        SPLIT=tensor MODEL="$Q4" ./scripts/run-spec-placement.sh h31-2card-f16reduce none none default
    grep -E '^(label|h28-2card-mmvh1,|h31-)' results/h11-placement.csv | column -s, -t
    python3 scripts/compare-outputs.py h28-2card-mmvh1 h31-2card-f16reduce
fi

git add -A results logs scripts/run-h31-f16-reduce.sh
git commit -q -m "bench: H31 F16 AllReduce payload (fork P1) KLD gate and 16k prefill" \
    && { git push -q && echo pushed || echo "WARNING: push failed; commit is local" >&2; }
