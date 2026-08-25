#!/usr/bin/env bash
# Phase 1 engine baseline on UD-Q4_K_M, plus the -sm graph cells that the
# ik NCCL rebuild unblocks. Strictly sequential; run-bench.sh commits each cell.
#
# Order is deliberate. ik exists in this benchmark only for -sm graph, so its
# graph cell and its same-build layer baseline run first - that pair is the
# actual question. ik-with-NCCL on graph follows only to complete the H5 pair
# (same tree, same model, one build flag apart); it aborts in ~1 min, so it is
# nearly free. ik-with-NCCL on layer is deliberately NOT run - nothing depends
# on it.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL=/root/Qwen3.8-27B-UD-Q4_K_M.gguf

# engine      split  label
CELLS=(
  "ik-nonccl  graph  phase1-iknonccl-graph-q4km"
  "ik-nonccl  layer  phase1-iknonccl-layer-q4km"
  "ik         graph  phase1-ik-graph-q4km"
  "pflash     layer  phase1-pflash-layer-q4km"
  "buun       layer  phase1-buun-layer-q4km"
  "mainline   layer  phase1-mainline-layer-q4km"
)

for cell in "${CELLS[@]}"; do
    read -r engine split label <<<"$cell"
    for _ in $(seq 1 40); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 70 ]] && break
        echo "cooling: hottest card ${t}C"; sleep 20
    done
    echo "############ $label ############"
    ./scripts/run-bench.sh "$engine" "$MODEL" "$split" "$label"
    echo "############ $label exit=$? ############"
    sleep 15
done
echo "=== phase 1 complete ==="
