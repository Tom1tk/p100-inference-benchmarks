#!/usr/bin/env bash
# Sample GPU telemetry every INTERVAL seconds until killed.
# Output line format: HH:MM:SS idx, tempC, watts, util% | idx, tempC, watts, util%
# If ABORT_TEMP is set, any card at or above it kills the parent driver's llama-*
# children (live abort; at 83 C a P100 drops off the bus) and logs an ABORT line.
set -uo pipefail

INTERVAL="${INTERVAL:-5}"
ABORT_TEMP="${ABORT_TEMP:-}"

while true; do
    printf '%s %s\n' \
        "$(date +%H:%M:%S)" \
        "$(nvidia-smi --query-gpu=index,temperature.gpu,power.draw,utilization.gpu \
              --format=csv,noheader,nounits | tr '\n' '|')"
    if [[ -n "$ABORT_TEMP" ]]; then
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        if (( t >= ABORT_TEMP )); then
            echo "$(date +%H:%M:%S) ABORT: card at ${t}C >= ${ABORT_TEMP}C, killing llama processes of driver pid $PPID"
            pkill -P "$PPID" -f 'bin/llama-'
        fi
    fi
    sleep "$INTERVAL"
done
