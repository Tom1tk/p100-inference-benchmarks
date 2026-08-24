#!/usr/bin/env bash
# Phase 5: agentic web-build benchmark for one model version.
#
# Usage: ./scripts/run-web-bench.sh <engine> <model-path> <split-mode> <label> <index> [extra llama-server args...]
#   engine      pflash | buun | ik
#   split-mode  see METHODOLOGY.md -- support differs per engine
#   label       e.g. p5-buun-layer-q6k-mtp   (folder, systemd unit, and commit message all key off this)
#   index       0,1,2,... unique per model version. Every port derives from it, so
#               each site stays hosted after its run instead of fighting for :4000.
#
# Drives `pi` through the three prompts in prompts/web-bench.md against a
# llama-server hosting this model version, recording per-request prefill/decode
# throughput via scripts/web_bench_metrics.py.
#
# Commits and pushes on both success and failure. A failed run is data.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# --- Fixed parameters (see WEB_BENCH.md) ------------------------------------
NGL=99
THREADS=8
FA=1
CTK=f16
CTV=f16
CTX="${CTX:-32768}"                 # agentic transcripts run long; drop to 16384 if the server OOMs
STAGE_TIMEOUT="${STAGE_TIMEOUT:-3600}"
LOAD_TIMEOUT="${LOAD_TIMEOUT:-900}" # model load is 4-8 min; 15 min covers disk contention

ABORT_TEMP=83
PREFLIGHT_TEMP=70

# --- Arguments --------------------------------------------------------------
if [[ $# -lt 5 ]]; then
    sed -n '2,17p' "$0" >&2
    exit 2
fi

ENGINE="$1"; MODEL="$2"; SPLIT="$3"; LABEL="$4"; IDX="$5"; shift 5
EXTRA_ARGS=("$@")

[[ "$IDX" =~ ^[0-9]+$ ]] || { echo "ERROR: index must be a non-negative integer, got '$IDX'" >&2; exit 2; }

case "$ENGINE" in
    pflash) BIN=/root/pflash-llama.cpp/build-cuda-p100/bin/llama-server ;;
    buun)   BIN=/root/buun-llama-cpp/build-cuda-p100/bin/llama-server ;;
    ik)     BIN=/root/ik_llama.cpp/build/bin/llama-server ;;
    *)      echo "ERROR: unknown engine '$ENGINE' (expected pflash|buun|ik)" >&2; exit 2 ;;
esac

[[ -x "$BIN"   ]] || { echo "ERROR: engine binary not found: $BIN" >&2; exit 2; }
[[ -f "$MODEL" ]] || { echo "ERROR: model not found: $MODEL" >&2; exit 2; }
command -v pi >/dev/null || { echo "ERROR: pi not on PATH" >&2; exit 2; }

# --- Ports: one block per index, so every site stays up ---------------------
SITE_PORT=$((4000 + IDX))
SERVER_PORT=$((8100 + IDX))
PROXY_PORT=$((8200 + IDX))

for port in "$SITE_PORT" "$SERVER_PORT" "$PROXY_PORT"; do
    if ss -ltn 2>/dev/null | grep -q ":${port}\b"; then
        echo "ERROR: port $port is already in use - pick a different index" >&2
        ss -ltnp 2>/dev/null | grep ":${port}\b" >&2
        exit 1
    fi
done

# --- Paths ------------------------------------------------------------------
mkdir -p logs results/web sites prompts
SERVER_LOG="logs/${LABEL}.server.log"
AGENT_LOG="logs/${LABEL}.agent.log"
TEMPS="logs/${LABEL}.temps.log"
METRICS="results/web/${LABEL}.jsonl"
STAGE_TIMES="results/web/${LABEL}.stages.json"
SUMMARY="results/web/${LABEL}.json"
AGGREGATE="results/web-bench.csv"
SITE_DIR="sites/${LABEL}"
PI_DIR="${SITE_DIR}/.pi-agent"

rm -f "$METRICS" "${METRICS}.stage"
mkdir -p "$SITE_DIR" "$PI_DIR"

# --- Preflight --------------------------------------------------------------
echo "=== preflight ==="
GPU_COUNT=$(nvidia-smi -L | grep -c 'Tesla P100')
if [[ "$GPU_COUNT" -ne 2 ]]; then
    echo "ERROR: expected 2 Tesla P100s, found $GPU_COUNT" >&2
    exit 1
fi

MAX_IDLE=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
echo "GPUs: $GPU_COUNT | hottest card: ${MAX_IDLE}C | site :$SITE_PORT | server :$SERVER_PORT | proxy :$PROXY_PORT"
if [[ "$MAX_IDLE" -gt "$PREFLIGHT_TEMP" ]]; then
    echo "ERROR: card at ${MAX_IDLE}C exceeds preflight limit ${PREFLIGHT_TEMP}C - let it cool" >&2
    exit 1
fi

# --- Teardown ---------------------------------------------------------------
# The site's systemd unit is deliberately left running - hosting every model's
# site at once is the point of the per-index port scheme.
SERVER_PID=""; PROXY_PID=""; MONITOR_PID=""
cleanup() {
    for pid in "$PROXY_PID" "$SERVER_PID" "$MONITOR_PID"; do
        [[ -n "$pid" ]] && kill "$pid" 2>/dev/null
    done
}
trap cleanup EXIT

# --- Telemetry --------------------------------------------------------------
INTERVAL=5 ./scripts/gpu-monitor.sh > "$TEMPS" 2>/dev/null &
MONITOR_PID=$!

# --- llama-server -----------------------------------------------------------
# --jinja is required: pi drives the model through tool calls, and without the
# Jinja chat template the server won't parse or emit them.
SERVER_CMD=(
    "$BIN" -m "$MODEL"
    --host 127.0.0.1 --port "$SERVER_PORT"
    -ngl "$NGL" -fa "$FA" -t "$THREADS"
    -ctk "$CTK" -ctv "$CTV"
    -sm "$SPLIT" -c "$CTX"
    --jinja
    "${EXTRA_ARGS[@]}"
)

echo "=== starting server: $LABEL ==="
printf '%q ' CUDA_VISIBLE_DEVICES=0,1 "${SERVER_CMD[@]}"; echo
echo "NOTE: model load takes 4-8 min with no output. This is not a hang."

START=$(date +%s)
CUDA_VISIBLE_DEVICES=0,1 "${SERVER_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# Poll /health until the model is resident, or the server dies trying.
deadline=$(( $(date +%s) + LOAD_TIMEOUT ))
until curl -sf -m 5 "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: server exited during load - last 30 lines:" >&2
        tail -30 "$SERVER_LOG" >&2
        OUTCOME="FAILED(load)"; ELAPSED=$(( $(date +%s) - START ))
        break
    fi
    if [[ $(date +%s) -gt $deadline ]]; then
        echo "ERROR: server not healthy after ${LOAD_TIMEOUT}s" >&2
        OUTCOME="FAILED(timeout)"; ELAPSED=$(( $(date +%s) - START ))
        break
    fi
    sleep 10
done

# --- Agent stages -----------------------------------------------------------
declare -A STAGE_SECONDS=()
if [[ -z "${OUTCOME:-}" ]]; then
    LOAD_SECONDS=$(( $(date +%s) - START ))
    echo "=== server healthy after ${LOAD_SECONDS}s ==="

    python3 ./scripts/web_bench_metrics.py proxy \
        --listen "$PROXY_PORT" --upstream "127.0.0.1:${SERVER_PORT}" --out "$METRICS" \
        >> "$SERVER_LOG" 2>&1 &
    PROXY_PID=$!
    sleep 2

    # Isolated pi config: never touch the operator's ~/.pi/agent settings.
    cat > "${PI_DIR}/models.json" <<EOF
{
  "providers": {
    "webbench": {
      "baseUrl": "http://127.0.0.1:${PROXY_PORT}/v1",
      "api": "openai-completions",
      "apiKey": "none",
      "models": [ { "id": "${LABEL}", "name": "${LABEL}" } ]
    }
  }
}
EOF

    OUTCOME="ok"
    for STAGE in 1 2 3; do
        PROMPT=$(awk "/<!-- STAGE ${STAGE} -->/{f=1;next} /<!-- END STAGE ${STAGE} -->/{f=0} f" \
                     prompts/web-bench.md \
                 | sed -e "s|{{MODEL_NAME}}|${LABEL}|g" -e "s|{{PORT}}|${SITE_PORT}|g")
        if [[ -z "${PROMPT// }" ]]; then
            echo "ERROR: stage ${STAGE} prompt is empty - check prompts/web-bench.md markers" >&2
            OUTCOME="FAILED(prompt)"
            break
        fi

        echo "$STAGE" > "${METRICS}.stage"
        echo "=== stage ${STAGE} ==="
        S0=$(date +%s)
        # --continue keeps stages 2 and 3 in the same session, so the agent sees
        # the site it already built rather than rediscovering it.
        PI_ARGS=(--provider webbench --model "$LABEL" --session-dir "${PI_DIR}/sessions" -p)
        [[ "$STAGE" -gt 1 ]] && PI_ARGS=(--continue "${PI_ARGS[@]}")

        ( cd "$SITE_DIR" && PI_OFFLINE=1 PI_CODING_AGENT_DIR="${REPO}/${PI_DIR}" \
            timeout "$STAGE_TIMEOUT" pi "${PI_ARGS[@]}" "$PROMPT" ) \
            >> "$AGENT_LOG" 2>&1
        STATUS=$?
        STAGE_SECONDS[$STAGE]=$(( $(date +%s) - S0 ))
        echo "stage ${STAGE}: ${STAGE_SECONDS[$STAGE]}s (exit ${STATUS})"

        if [[ "$STATUS" -ne 0 ]]; then
            echo "ERROR: stage ${STAGE} exited ${STATUS} (124 = hit the ${STAGE_TIMEOUT}s timeout)" >&2
            tail -20 "$AGENT_LOG" >&2
            OUTCOME="FAILED(stage${STAGE})"
            break
        fi
    done
    ELAPSED=$(( $(date +%s) - START ))
fi

kill "$PROXY_PID" 2>/dev/null; PROXY_PID=""
kill "$SERVER_PID" 2>/dev/null; SERVER_PID=""
kill "$MONITOR_PID" 2>/dev/null; MONITOR_PID=""
trap - EXIT

# --- Telemetry summary ------------------------------------------------------
PEAK_TEMP=$(awk '{
    sub(/^[0-9:]+ /, "")
    n = split($0, cards, "|")
    for (i = 1; i <= n; i++) {
        if (split(cards[i], f, ",") >= 2) { t = f[2] + 0; if (t > max) max = t }
    }
} END { print (max ? max : "n/a") }' "$TEMPS")

echo "=== finished in ${ELAPSED}s | ${OUTCOME} | peak ${PEAK_TEMP}C ==="
if [[ "$PEAK_TEMP" != "n/a" && "$PEAK_TEMP" -ge "$ABORT_TEMP" ]]; then
    echo "WARNING: peak ${PEAK_TEMP}C reached the ${ABORT_TEMP}C abort threshold." >&2
    echo "WARNING: treat this run's numbers as throttled and record it in RUNLOG.md." >&2
fi

# --- Summarize --------------------------------------------------------------
{
    printf '{'
    sep=""
    for stage in "${!STAGE_SECONDS[@]}"; do
        printf '%s"%s": %s' "$sep" "$stage" "${STAGE_SECONDS[$stage]}"
        sep=", "
    done
    printf '}\n'
} > "$STAGE_TIMES"

python3 ./scripts/web_bench_metrics.py summarize \
    --metrics "$METRICS" --stage-times "$STAGE_TIMES" \
    --label "$LABEL" --engine "$ENGINE" --model "$MODEL" \
    --site-port "$SITE_PORT" --total-seconds "$ELAPSED" \
    --out "$SUMMARY" --csv "$AGGREGATE"

echo
echo "Site should now be live at http://localhost:${SITE_PORT} (systemd unit ${LABEL}.service)"
systemctl is-active "${LABEL}.service" 2>/dev/null || echo "NOTE: ${LABEL}.service is not active - the agent may not have completed stage 1"

# --- Commit and push (mandatory, pass or fail) ------------------------------
git add -A
if git diff --cached --quiet; then
    echo "nothing to commit"
else
    git commit -q -m "webbench: ${LABEL} — ${OUTCOME} (${ELAPSED}s, peak ${PEAK_TEMP}C, :${SITE_PORT})"
    git push -q && echo "pushed" || echo "WARNING: push failed; commit is local" >&2
fi

echo
echo "Next: update RESULTS.md Phase 5 table and WEB_BENCH.md's port registry (see RUNBOOK.md section 4)."

[[ "$OUTCOME" == "ok" ]]
