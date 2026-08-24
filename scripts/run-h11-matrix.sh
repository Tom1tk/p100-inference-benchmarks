#!/usr/bin/env bash
# H11 driver: every (drafter x placement) cell, plus the no-drafter control.
# Strictly sequential - two arms sharing the GPUs would invalidate both.
# A failed arm is recorded and the matrix continues; a failed run is data.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MTP=/root/mtp-Qwen3.8-27B-Q4_0.gguf
DF4=/root/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
DF8=/root/Qwen3.8-27B-DFlash2-Q8_0.gguf

# label                  spec-type      drafter  placement
ARMS=(
  "h11-none-control      none           none     default"
  "h11-mtp-cuda0         draft-mtp      $MTP     CUDA0"
  "h11-mtp-cuda1         draft-mtp      $MTP     CUDA1"
  "h11-df2q4-default     draft-dflash   $DF4     default"
  "h11-df2q4-cuda0       draft-dflash   $DF4     CUDA0"
  "h11-df2q4-cuda1       draft-dflash   $DF4     CUDA1"
  "h11-df2q8-default     draft-dflash   $DF8     default"
  "h11-df2q8-cuda0       draft-dflash   $DF8     CUDA0"
  "h11-df2q8-cuda1       draft-dflash   $DF8     CUDA1"
)

for arm in "${ARMS[@]}"; do
    read -r label spec drafter placement <<<"$arm"
    # Preflight enforces <=70C; give the cards a chance to reach it rather than
    # failing the arm outright.
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 70 ]] && break
        echo "cooling: hottest card ${t}C"; sleep 20
    done
    ./scripts/run-spec-placement.sh "$label" "$spec" "$drafter" "$placement"
    echo "--- $label done (exit $?) ---"
    sleep 15   # let VRAM actually free before the next load
done
echo "=== matrix complete ==="
column -s, -t results/h11-placement.csv
