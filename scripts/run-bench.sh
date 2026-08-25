#!/usr/bin/env bash
# Run one benchmark cell: preflight -> telemetry -> benchmark -> log -> commit+push.
#
# Usage: ./scripts/run-bench.sh <engine> <model-path> <split-mode> <label> [extra llama-bench args...]
#   engine      pflash | buun | ik | ik-nonccl | mainline
#   split-mode  see METHODOLOGY.md -- support differs per engine
#   label       e.g. phase1-pflash-layer-q6k  (used for filenames and commit message)
#
# Commits and pushes on both success and failure. A failed run is data.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# --- Fixed parameters (see METHODOLOGY.md section 6) -------------------------
NGL=99
THREADS=8
FA=1
CTK=f16
CTV=f16
REPS=3
PROMPT_DEPTHS="0,2048,4096,8192,16384"
GEN_TOKENS=128

ABORT_TEMP=83     # hard abort threshold, degrees C
PREFLIGHT_TEMP=70 # refuse to start if a card is hotter than this

# --- Arguments --------------------------------------------------------------
if [[ $# -lt 4 ]]; then
    sed -n '2,12p' "$0" >&2
    exit 2
fi

ENGINE="$1"; MODEL="$2"; SPLIT="$3"; LABEL="$4"; shift 4
EXTRA_ARGS=("$@")

case "$ENGINE" in
    pflash) BIN=/root/pflash-llama.cpp/build-cuda-p100/bin/llama-bench ;;
    buun)   BIN=/root/buun-llama-cpp/build-cuda-p100/bin/llama-bench ;;
    ik)     BIN=/root/ik_llama.cpp/build/bin/llama-bench ;;
    ik-nonccl) BIN=/root/ik_llama.cpp/build-nonccl/bin/llama-bench ;;   # H5: same tree, -DGGML_NCCL=OFF
    mainline) BIN=/root/dflash2-llama.cpp/build-cuda-p100/bin/llama-bench ;;
    *)      echo "ERROR: unknown engine '$ENGINE' (expected pflash|buun|ik|ik-nonccl|mainline)" >&2; exit 2 ;;
esac

[[ -x "$BIN"   ]] || { echo "ERROR: engine binary not found: $BIN" >&2; exit 2; }
[[ -f "$MODEL" ]] || { echo "ERROR: model not found: $MODEL" >&2; exit 2; }

mkdir -p logs results

LOG="logs/${LABEL}.log"
CSV="results/raw/${LABEL}.csv"
TEMPS="logs/${LABEL}.temps.log"
AGGREGATE="results/all-results.csv"
mkdir -p results/raw

# --- Preflight --------------------------------------------------------------
echo "=== preflight ==="
GPU_COUNT=$(nvidia-smi -L | grep -c 'Tesla P100')
if [[ "$GPU_COUNT" -ne 2 ]]; then
    echo "ERROR: expected 2 Tesla P100s, found $GPU_COUNT" >&2
    nvidia-smi -L >&2
    exit 1
fi

MAX_IDLE=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
echo "GPUs: $GPU_COUNT | hottest card: ${MAX_IDLE}C"
if [[ "$MAX_IDLE" -gt "$PREFLIGHT_TEMP" ]]; then
    echo "ERROR: card at ${MAX_IDLE}C exceeds preflight limit ${PREFLIGHT_TEMP}C - let it cool" >&2
    exit 1
fi

nvidia-smi --query-gpu=index,power.limit,persistence_mode --format=csv,noheader

# --- Telemetry --------------------------------------------------------------
INTERVAL=5 ./scripts/gpu-monitor.sh > "$TEMPS" 2>/dev/null &
MONITOR_PID=$!
# shellcheck disable=SC2064
trap "kill $MONITOR_PID 2>/dev/null" EXIT

# --- Benchmark --------------------------------------------------------------
CMD=(
    "$BIN" -m "$MODEL"
    -ngl "$NGL" -fa "$FA" -t "$THREADS"
    -ctk "$CTK" -ctv "$CTV"
    -sm "$SPLIT"
    -p "$PROMPT_DEPTHS" -n "$GEN_TOKENS" -r "$REPS"
    -o csv
    "${EXTRA_ARGS[@]}"
)

echo "=== running: $LABEL ==="
printf '%q ' CUDA_VISIBLE_DEVICES=0,1 "${CMD[@]}"; echo
echo "NOTE: model load takes 4-8 min with no output. This is not a hang."

START=$(date +%s)
CUDA_VISIBLE_DEVICES=0,1 "${CMD[@]}" > "$CSV" 2> "$LOG"
STATUS=$?
ELAPSED=$(( $(date +%s) - START ))

kill "$MONITOR_PID" 2>/dev/null
trap - EXIT

# --- Telemetry summary ------------------------------------------------------
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
    echo "WARNING: treat this run's numbers as throttled and record it in RUNLOG.md." >&2
fi

if [[ "$STATUS" -eq 0 && -s "$CSV" ]]; then
    OUTCOME="ok"
    # Aggregate: header once, then data rows prefixed with the run label.
    if [[ ! -f "$AGGREGATE" ]]; then
        { printf 'label,'; head -1 "$CSV"; } > "$AGGREGATE"
    fi
    tail -n +2 "$CSV" | while IFS= read -r row; do
        printf '%s,%s\n' "$LABEL" "$row" >> "$AGGREGATE"
    done
    echo "--- results ---"
    cat "$CSV"
else
    OUTCOME="FAILED"
    echo "--- run failed, last 20 log lines ---" >&2
    tail -20 "$LOG" >&2
fi

# --- Commit and push (mandatory, pass or fail) ------------------------------
git add -A
if git diff --cached --quiet; then
    echo "nothing to commit"
else
    git commit -q -m "bench: ${LABEL} — ${OUTCOME} (${ELAPSED}s, peak ${PEAK_TEMP}C)"
    git push -q && echo "pushed" || echo "WARNING: push failed; commit is local" >&2
fi

echo
echo "Next: update RESULTS.md, and HYPOTHESES.md if this bears on H1-H7 (see RUNBOOK.md section 4)."

exit "$STATUS"
