#!/usr/bin/env bash
# Sample GPU telemetry every INTERVAL seconds until killed.
# Output line format: HH:MM:SS idx, tempC, watts, util% | idx, tempC, watts, util%
set -uo pipefail

INTERVAL="${INTERVAL:-5}"

while true; do
    printf '%s %s\n' \
        "$(date +%H:%M:%S)" \
        "$(nvidia-smi --query-gpu=index,temperature.gpu,power.draw,utilization.gpu \
              --format=csv,noheader,nounits | tr '\n' '|')"
    sleep "$INTERVAL"
done
