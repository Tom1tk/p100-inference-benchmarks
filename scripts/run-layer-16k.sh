#!/usr/bin/env bash
# Close the matched-pair gap at depth: Phase 2 has tensor control + tensor MTP
# at 16k, but the only layer-split 16k rows use DFlash2. Without a layer
# control and a layer+MTP cell, the tensor-vs-layer delta at depth rests on a
# different drafter. These two cells make it drafter-matched.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

while pgrep -f 'run-h10-p2p.sh|llama-server|llama-bench' >/dev/null; do sleep 30; done

export SPLIT=layer REPS=3 N_PREDICT=400 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt

cool() {
    for _ in $(seq 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 70 ]] && return 0
        echo "cooling: hottest ${t}C"; sleep 20
    done
}

run() { cool; echo "########## $1"; ./scripts/run-spec-placement.sh "$@" 2>&1 | tail -8; }

run p2-layer-none-16k none      none                            default
run p2-layer-mtp-16k  draft-mtp /root/mtp-Qwen3.8-27B-Q4_0.gguf default

echo "=== layer 16k pair complete ==="
