#!/usr/bin/env bash
# H28 (fork D2): Pascal half2 matvec. KLD gate first, then 16k decode arms vs banked H27/D5 controls.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FORKBIN=/root/llama.cpp-Pascal/build-cuda-p100/bin
Q4=/root/Qwen3.8-27B-UD-Q4_K_M.gguf; IQ3=/root/Qwen3.8-27B-UD-IQ3_S.gguf
KLDIR=/root/h28-kld; mkdir -p "$KLDIR" results/h28 logs
CSV=results/h28-kld.csv

cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 60 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20
    done
}
ppl() {  # label mmvh ub extra-args...
    local label=$1 mmvh=$2 ub=$3; shift 3
    cool
    ABORT_TEMP=80 INTERVAL=2 ./scripts/gpu-monitor.sh > "logs/h28-$label-temps.log" 2>/dev/null &
    local mon=$!
    GGML_CUDA_PASCAL_MMVH=$mmvh "$FORKBIN/llama-perplexity" -m "$Q4" -f prompts/h26-100k-prompt.txt \
        -c 512 --chunks 8 -b 512 -ub "$ub" -ngl 99 -fa 1 -sm layer "$@" > "logs/h28-$label.log" 2>&1
    local rc=$?
    kill $mon
    echo "--- $label rc=$rc, peak $(tr '|' '\n' < "logs/h28-$label-temps.log" | awk -F', ' 'NF>=3{print $2}' | sort -n | tail -1)C ---"
    return $rc
}
kld_row() {  # label
    local f="logs/h28-$1.log"
    printf '%s,%s,%s,%s\n' "$1" \
        "$(grep -m1 'Mean    KLD:' "$f" | awk '{print $3}')" \
        "$(grep -m1 '99.0%   KLD:' "$f" | awk '{print $3}')" \
        "$(grep -m1 'Same top p:' "$f" | awk '{print $4}')" >> "$CSV"
}

echo "label,mean_kld,p99_kld,same_top_pct" > "$CSV"
ppl base-mmvq-ub8 0 8 --kl-divergence-base "$KLDIR/base.kld" || exit 1
ppl ref-mmq-ub512 0 512 --kl-divergence-base "$KLDIR/base.kld" --kl-divergence && kld_row ref-mmq-ub512
ppl mmvh-ub8 1 8 --kl-divergence-base "$KLDIR/base.kld" --kl-divergence && kld_row mmvh-ub8
rm -f "$KLDIR/base.kld"
column -s, -t "$CSV"

GATE=$(python3 - "$CSV" <<'PY'
import csv, sys
r = {x["label"]: x for x in csv.DictReader(open(sys.argv[1]))}
ref, m = r["ref-mmq-ub512"], r["mmvh-ub8"]
ok = float(m["mean_kld"]) <= 2*float(ref["mean_kld"]) and float(m["same_top_pct"]) >= float(ref["same_top_pct"]) - 0.5
print("PASS" if ok else "FAIL")
PY
)
echo "KLD gate: $GATE"

if [[ "$GATE" == PASS ]]; then
    export GGML_CUDA_ALLREDUCE=none GGML_CUDA_PASCAL_MMVH=1 REPS=2 N_PREDICT=400 CTX=16384 \
           PROMPT_FILE=prompts/h11-depth-prompt.txt BATCH=2048 UB=2048 BIN=$FORKBIN/llama-server
    arm() { cool; "$@"; echo "--- done (exit $?) ---"; sleep 15; }
    arm env SPLIT=tensor MODEL="$Q4"  ./scripts/run-spec-placement.sh h28-2card-mmvh1 none none default
    arm env SPLIT=none   MODEL="$IQ3" ./scripts/run-spec-placement.sh h28-1card-mmvh1 none none default
    arm env SPLIT=tensor GGML_CUDA_P2P=1 ./scripts/run-spec-placement.sh h28-2card-mmvh1-mtp-p2p draft-mtp \
        /root/mtp-Qwen3.8-27B-Q4_0.gguf default
    echo "=== H28 results ==="
    grep -E '^(label|h27-.*graphs0|d5-p2p|h28-)' results/h11-placement.csv | column -s, -t
fi

git add -A results logs HYPOTHESES.md scripts/run-h28-mmvh.sh
git commit -q -m "bench: H28 Pascal half2 matvec (fork D2) KLD gate and 16k decode arms" \
    && { git push -q && echo pushed || echo "WARNING: push failed; commit is local" >&2; }
