#!/usr/bin/env bash
# H14 (Phase 7 cell 3) — quick ubatch sweep at low context.
#
# One llama-bench invocation so the 5-8 min model load is paid ONCE and the
# whole ub sweep runs inside it. Prefill only (-n 0): H14 is a prefill claim,
# and skipping tg keeps this inside the 30 min budget.
#
# Every prefill number in this repo was taken at the default -ub 512 and has
# never been swept. See HYPOTHESES.md H14 and METHODOLOGY "Prefill is measured
# at length, not depth".

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

BIN=/root/dflash2-rebased/build-cuda-p100/bin/llama-bench
MODEL=/root/Qwen3.8-27B-UD-Q4_K_M.gguf
LABEL=h14-ubatch-sweep-q4km

UBATCHES=128,256,512,1024,2048
PROMPTS=2048,4096
REPS=2

ABORT_TEMP=83
PREFLIGHT_TEMP=70

LOG="logs/${LABEL}.log"
CSV="results/raw/${LABEL}.csv"
TEMPS="logs/${LABEL}.temps.log"
mkdir -p logs results/raw

echo "=== preflight ==="
MAX_IDLE=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
echo "hottest card: ${MAX_IDLE}C"
if [[ "$MAX_IDLE" -gt "$PREFLIGHT_TEMP" ]]; then
    echo "ERROR: card at ${MAX_IDLE}C exceeds preflight limit ${PREFLIGHT_TEMP}C - let it cool" >&2
    exit 1
fi
nvidia-smi --query-gpu=index,power.limit --format=csv,noheader

INTERVAL=5 ./scripts/gpu-monitor.sh > "$TEMPS" 2>/dev/null &
MONITOR_PID=$!
# shellcheck disable=SC2064
trap "kill $MONITOR_PID 2>/dev/null" EXIT

echo "=== running: $LABEL ==="
echo "ub=${UBATCHES}  p=${PROMPTS}  r=${REPS}  -sm tensor  ALLREDUCE=none"
echo "NOTE: model load takes 4-8 min with no output. This is not a hang."

START=$(date +%s)
CUDA_VISIBLE_DEVICES=0,1 GGML_CUDA_ALLREDUCE=none "$BIN" \
    -m "$MODEL" \
    -ngl 99 -fa 1 -t 8 \
    -ctk f16 -ctv f16 \
    -sm tensor \
    -ub "$UBATCHES" \
    -p "$PROMPTS" -n 0 -r "$REPS" \
    -o csv > "$CSV" 2> "$LOG"
STATUS=$?
ELAPSED=$(( $(date +%s) - START ))

kill "$MONITOR_PID" 2>/dev/null
trap - EXIT

PEAK_TEMP=$(awk '{
    sub(/^[0-9:]+ /, "")
    n = split($0, cards, "|")
    for (i = 1; i <= n; i++) {
        if (split(cards[i], f, ",") >= 2) { t = f[2] + 0; if (t > max) max = t }
    }
} END { print (max ? max : "n/a") }' "$TEMPS")

echo "=== finished in ${ELAPSED}s | exit ${STATUS} | peak ${PEAK_TEMP}C ==="
if [[ "$PEAK_TEMP" != "n/a" && "$PEAK_TEMP" -ge "$ABORT_TEMP" ]]; then
    echo "WARNING: peak ${PEAK_TEMP}C reached the ${ABORT_TEMP}C abort threshold." >&2
fi

if [[ "$STATUS" -eq 0 && -s "$CSV" ]]; then
    OUTCOME="ok"
    echo "--- results ---"
    python3 - "$CSV" <<'PYEOF'
import csv,sys
rd=csv.reader(open(sys.argv[1])); hdr=next(rd); H={n:i for i,n in enumerate(hdr)}
print(f"{'n_ubatch':>9} {'n_batch':>8} {'n_prompt':>9} {'t/s':>9} {'stddev':>8}")
for r in rd:
    if len(r)!=len(hdr): continue
    print(f"{r[H['n_ubatch']]:>9} {r[H['n_batch']]:>8} {r[H['n_prompt']]:>9} "
          f"{float(r[H['avg_ts']]):9.2f} {float(r[H['stddev_ts']]):8.2f}")
PYEOF
else
    OUTCOME="FAILED"
    echo "--- run failed, last 20 log lines ---" >&2
    tail -20 "$LOG" >&2
fi

git add -A
if git diff --cached --quiet; then
    echo "nothing to commit"
else
    git commit -q -m "bench: ${LABEL} — ${OUTCOME} (${ELAPSED}s, peak ${PEAK_TEMP}C)"
    git push -q && echo "pushed" || echo "WARNING: push failed; commit is local" >&2
fi

exit "$STATUS"
