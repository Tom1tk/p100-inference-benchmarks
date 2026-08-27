#!/usr/bin/env bash
# H11: does draft-model placement change decode throughput on a two-card split?
#
# Usage: ./scripts/run-spec-placement.sh <label> <spec-type> <drafter-path|none> <placement>
#   spec-type   draft-mtp | draft-dflash | none
#   drafter     path to drafter GGUF, or 'none' for the no-drafter control
#   placement   default | CUDA0 | CUDA1     (default = no -devd, drafter splits over both)
#
# Everything except the drafter and its placement is pinned, so a difference in
# decode t/s is attributable to placement. Acceptance rate is recorded as a
# control: placement must NOT change it, so if it moves, something else did too.
#
# Appends one CSV row per repetition to results/h11-placement.csv.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# --- Fixed parameters -------------------------------------------------------
MODEL="${MODEL:-/root/Qwen3.8-27B-UD-Q4_K_M.gguf}"
# Phase 1 selected -sm tensor, so SPLIT is now an env knob rather than a
# constant. Tensor split needs GGML_CUDA_ALLREDUCE=internal (or none) -- the
# NCCL default aborts. See H5/H10.
SPLIT="${SPLIT:-layer}"
BIN="${BIN:-/root/dflash2-rebased/build-cuda-p100/bin/llama-server}"
NGL=99
THREADS=8
FA=1
CTX="${CTX:-4096}"
N_PREDICT="${N_PREDICT:-400}"
# buun defaults KV to vbr, not f16 -- always pin these. See METHODOLOGY.
CTK="${CTK:-f16}"
CTV="${CTV:-f16}"
REPS="${REPS:-3}"
# H24: -ub/-b are server launch params, so a ubatch arm needs a reload.
# Empty means "leave llama-server at its defaults" (-b 2048 -ub 512).
BATCH="${BATCH:-}"
UB="${UB:-}"
# H26: at 100k the default n_slots=4 is a needless allocation risk, and one
# request can run for ~8 min, well past the old fixed 900s curl timeout.
NP="${NP:-}"
REQ_TIMEOUT="${REQ_TIMEOUT:-900}"
PORT=8300
LOAD_TIMEOUT=900
ABORT_TEMP=83
PREFLIGHT_TEMP=70

# PROMPT_FILE lets a run decode at real context depth. -c alone only sizes the
# KV allocation; depth comes from how many tokens are actually in the cache.
PROMPT_FILE="${PROMPT_FILE:-}"
if [[ -n "$PROMPT_FILE" ]]; then
    [[ -f "$PROMPT_FILE" ]] || { echo "ERROR: PROMPT_FILE not found: $PROMPT_FILE" >&2; exit 2; }
    PROMPT=$(cat "$PROMPT_FILE")
else
    PROMPT='Write a detailed technical explanation of how HTTP/2 multiplexing works. Cover streams, frames, flow control, and HPACK header compression, and explain how it differs from HTTP/1.1 pipelining.'
fi

# --- Arguments --------------------------------------------------------------
[[ $# -eq 4 ]] || { sed -n '2,14p' "$0" >&2; exit 2; }
LABEL="$1"; SPEC_TYPE="$2"; DRAFTER="$3"; PLACEMENT="$4"

[[ -x "$BIN"   ]] || { echo "ERROR: engine binary not found: $BIN" >&2; exit 2; }
[[ -f "$MODEL" ]] || { echo "ERROR: model not found: $MODEL" >&2; exit 2; }
[[ "$DRAFTER" == none || -f "$DRAFTER" ]] || { echo "ERROR: drafter not found: $DRAFTER" >&2; exit 2; }

mkdir -p logs results/h11
SERVER_LOG="logs/${LABEL}.server.log"
TEMPS="logs/${LABEL}.temps.log"
RAW="results/h11/${LABEL}.jsonl"
CSV="results/h11-placement.csv"
: > "$RAW"

if [[ ! -f "$CSV" ]]; then
    echo "label,spec_type,drafter,placement,rep,decode_tps,prefill_tps,n_decoded,draft_n,draft_accepted,accept_pct,gpu0_mib,gpu1_mib,peak_temp,load_s,outcome" > "$CSV"
fi

row() {  # rep decode prefill ndec draftn draftacc acc g0 g1 peak load outcome
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$LABEL" "$SPEC_TYPE" "$(basename "$DRAFTER")" "$PLACEMENT" "$@" >> "$CSV"
}

# --- Preflight --------------------------------------------------------------
if ss -ltn 2>/dev/null | grep -q ":${PORT}\b"; then
    echo "ERROR: port $PORT already in use" >&2; exit 1
fi
MAX_IDLE=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
if [[ "$MAX_IDLE" -gt "$PREFLIGHT_TEMP" ]]; then
    echo "ERROR: card at ${MAX_IDLE}C exceeds preflight limit ${PREFLIGHT_TEMP}C - let it cool" >&2
    exit 1
fi
echo "=== $LABEL === (idle ${MAX_IDLE}C)"

SERVER_PID=""; MONITOR_PID=""
cleanup() {
    for pid in "$SERVER_PID" "$MONITOR_PID"; do [[ -n "$pid" ]] && kill "$pid" 2>/dev/null; done
    # llama-server can take a few seconds to release VRAM; the next arm's
    # preflight would otherwise see a busy card.
    [[ -n "$SERVER_PID" ]] && wait "$SERVER_PID" 2>/dev/null
}
trap cleanup EXIT

INTERVAL=5 ./scripts/gpu-monitor.sh > "$TEMPS" 2>/dev/null &
MONITOR_PID=$!

# --- Server -----------------------------------------------------------------
SERVER_CMD=(
    "$BIN" -m "$MODEL"
    --host 127.0.0.1 --port "$PORT"
    -ngl "$NGL" -fa "$FA" -t "$THREADS"
    -ctk "$CTK" -ctv "$CTV"
    -sm "$SPLIT" -c "$CTX"
)
[[ -n "$BATCH" ]] && SERVER_CMD+=( -b "$BATCH" )
[[ -n "$UB"    ]] && SERVER_CMD+=( -ub "$UB" )
[[ -n "$NP"    ]] && SERVER_CMD+=( -np "$NP" )
if [[ "$DRAFTER" != none ]]; then
    SERVER_CMD+=( --spec-type "$SPEC_TYPE" -md "$DRAFTER" )
    [[ "$PLACEMENT" != default ]] && SERVER_CMD+=( -devd "$PLACEMENT" )
fi

printf '%q ' CUDA_VISIBLE_DEVICES=0,1 "${SERVER_CMD[@]}"; echo
START=$(date +%s)
CUDA_VISIBLE_DEVICES=0,1 "${SERVER_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

OUTCOME=""
deadline=$(( $(date +%s) + LOAD_TIMEOUT ))
until curl -sf -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: server exited during load - last 25 lines:" >&2
        tail -25 "$SERVER_LOG" >&2
        OUTCOME="FAILED(load)"; break
    fi
    if [[ $(date +%s) -gt $deadline ]]; then
        echo "ERROR: not healthy after ${LOAD_TIMEOUT}s" >&2
        OUTCOME="FAILED(timeout)"; break
    fi
    sleep 10
done
LOAD_S=$(( $(date +%s) - START ))

if [[ -n "$OUTCOME" ]]; then
    row 0 "" "" "" "" "" "" "" "" "" "$LOAD_S" "$OUTCOME"
    exit 1
fi

read -r GPU0 GPU1 <<<"$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
echo "loaded in ${LOAD_S}s | VRAM ${GPU0}/${GPU1} MiB"

# --- Repetitions ------------------------------------------------------------
for rep in $(seq 1 "$REPS"); do
    TEMP_NOW=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
    if [[ "$TEMP_NOW" -ge "$ABORT_TEMP" ]]; then
        echo "ABORT: card at ${TEMP_NOW}C >= ${ABORT_TEMP}C" >&2
        row "$rep" "" "" "" "" "" "" "$GPU0" "$GPU1" "$TEMP_NOW" "$LOAD_S" "ABORTED(temp)"
        exit 1
    fi

    # -s not -sf: on an HTTP error we want the body, which carries the reason.
    RESP=$(curl -s -m "$REQ_TIMEOUT" "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(python3 -c '
import json,sys
print(json.dumps({"messages":[{"role":"user","content":sys.argv[1]}],
 "temperature":0,"top_k":1,"seed":42,"max_tokens":int(sys.argv[2]),"cache_prompt":False,
 "timings_per_token":False}))' "$PROMPT" "$N_PREDICT")") || {
        echo "ERROR: request rep $rep failed (curl)" >&2
        row "$rep" "" "" "" "" "" "" "$GPU0" "$GPU1" "" "$LOAD_S" "FAILED(request)"
        continue
    }

    echo "$RESP" >> "$RAW"
    if ! grep -q '"timings"' <<<"$RESP"; then
        echo "ERROR: rep $rep returned no timings - server said:" >&2
        python3 -c 'import json,sys; print("  "+str(json.loads(sys.argv[1]).get("error","<unparseable>")))' "$RESP" >&2 2>/dev/null \
            || echo "  ${RESP:0:400}" >&2
        row "$rep" "" "" "" "" "" "" "$GPU0" "$GPU1" "" "$LOAD_S" "FAILED(request)"
        continue
    fi
    PEAK=$(awk -F'[ ,|]+' '{for(i=3;i<=NF;i+=4) if($i+0>m) m=$i+0} END{print m+0}' "$TEMPS")
    eval "$(python3 - "$RESP" <<'PY'
import json,sys
t = json.loads(sys.argv[1]).get("timings") or {}
dn, da = t.get("draft_n", 0) or 0, t.get("draft_n_accepted", 0) or 0
print(f'DEC={t.get("predicted_per_second",0):.2f}')
print(f'PRE={t.get("prompt_per_second",0):.2f}')
print(f'NDEC={t.get("predicted_n",0)}')
print(f'DN={dn}'); print(f'DA={da}')
print(f'ACC={(100.0*da/dn):.1f}' if dn else 'ACC=')
PY
)"
    echo "  rep${rep}: decode ${DEC} t/s | prefill ${PRE} t/s | accept ${DA}/${DN} = ${ACC:-n/a}% | peak ${PEAK}C"
    row "$rep" "$DEC" "$PRE" "$NDEC" "$DN" "$DA" "${ACC:-}" "$GPU0" "$GPU1" "$PEAK" "$LOAD_S" "ok"
done
