#!/usr/bin/env bash
# Phase 8 tier 2: one 100k haystack, 19 questions, on one serve config.
# Usage: [BIN=..] [MODEL=..] [SPEC=mtp|none] [SPLIT=tensor|none] [KV=q8_0|f16] [env GGML_CUDA_*] run-phase8-tier2.sh <label>
# Everything not listed is the frozen serve command (QUALITY-PLAN §0).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL=$1
BIN=${BIN:-/root/dflash2-rebased/build-cuda-p100/bin/llama-server}
MODEL=${MODEL:-/root/Qwen3.8-27B-UD-Q4_K_M.gguf}
SPEC=${SPEC:-mtp}; SPLIT=${SPLIT:-tensor}; PORT=8301
DRAFT=(); [[ $SPEC == mtp ]] && DRAFT=(--spec-type draft-mtp -md /root/mtp-Qwen3.8-27B-Q4_0.gguf)
mkdir -p logs
ABORT_TEMP=80 INTERVAL=2 ./scripts/gpu-monitor.sh > "logs/p8-$LABEL-temps.log" 2>/dev/null &
MON=$!
"$BIN" -m "$MODEL" "${DRAFT[@]}" -ngl 99 -fa 1 -t 8 -ctk "${KV:-q8_0}" -ctv "${KV:-q8_0}" -b 2048 -ub 2048 \
    -c 100000 -np 1 -sm "$SPLIT" --host 127.0.0.1 --port $PORT > "logs/p8-$LABEL-server.log" 2>&1 &
SRV=$!
for _ in $(seq 1 180); do
    curl -sf localhost:$PORT/health >/dev/null && break
    kill -0 $SRV 2>/dev/null || { echo "server died"; break; }
    kill -0 $MON 2>/dev/null || { echo "temp abort"; break; }
    sleep 5
done
RC=1
curl -sf localhost:$PORT/health >/dev/null && { python3 scripts/phase8-ask.py "$LABEL" $PORT; RC=$?; }
kill $SRV $MON 2>/dev/null; wait $SRV 2>/dev/null
echo "--- $LABEL rc=$RC peak $(tr '|' '\n' < "logs/p8-$LABEL-temps.log" | awk -F', ' 'NF>=3{print $2}' | sort -n | tail -1)C, VRAM $(grep -m2 -o 'CUDA[01] model buffer size = *[0-9.]* MiB' "logs/p8-$LABEL-server.log" | tr '\n' ' ') ---"
exit $RC
