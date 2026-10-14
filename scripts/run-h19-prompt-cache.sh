#!/usr/bin/env bash
# H19: does llama-server's prompt cache skip re-prefill on agentic turn 2+?
#
# One load of the serve command, flags left at their defaults (cache-ram 8192,
# ctx-checkpoints 32, cache-idle-slots on). The hybrid model can't roll GDN state
# back, so reuse hinges on the checkpoints the server takes at the last user
# message. The control is the same final prompt with cache_prompt=false, in the
# same load, so no second launch is needed.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LABEL=${LABEL:-h19-prompt-cache-defaults}
PORT=8310
BIN=/root/dflash2-rebased/build-cuda-p100/bin/llama-server
mkdir -p logs results
SERVER_LOG="logs/${LABEL}.server.log"; TEMPS="logs/${LABEL}.temps.log"

MAX_IDLE=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
[[ "$MAX_IDLE" -gt 70 ]] && { echo "card at ${MAX_IDLE}C, let it cool" >&2; exit 1; }

SERVER_PID=""; MONITOR_PID=""
trap 'kill $SERVER_PID $MONITOR_PID 2>/dev/null' EXIT
ABORT_TEMP=80 INTERVAL=5 ./scripts/gpu-monitor.sh > "$TEMPS" 2>/dev/null &
MONITOR_PID=$!

CUDA_VISIBLE_DEVICES=0,1 GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 \
"$BIN" -m /root/Qwen3.8-27B-UD-Q4_K_M.gguf --host 127.0.0.1 --port $PORT \
    -ngl 99 -fa 1 -t 8 -ctk q8_0 -ctv q8_0 -sm tensor -c 100000 -np 1 \
    -b 2048 -ub 2048 --jinja --spec-type draft-mtp -md /root/mtp-Qwen3.8-27B-Q4_0.gguf \
    "$@" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "loading (4-8 min, silent)..."
until curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null; do
    kill -0 $SERVER_PID 2>/dev/null || { echo "server died"; tail -30 "$SERVER_LOG"; exit 1; }
    sleep 10
done

python3 scripts/h19_client.py $PORT prompts/h11-depth-prompt.txt results/h19-prompt-cache.csv "$LABEL"
rc=$?
echo "peak temp: $(awk -F'|' '{for(i=1;i<=NF;i++){split($i,a,", "); if(a[2]+0>m)m=a[2]+0}} END{print m}' "$TEMPS")C"
exit $rc
