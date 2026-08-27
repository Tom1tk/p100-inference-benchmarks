#!/usr/bin/env bash
# H24: does -ub 2048 (H14's prefill win) cost decode at 16k?
#
# Two arms, identical in everything but -ub, both reloaded from scratch because
# -ub is a server launch parameter. The 512 arm is the control: it re-measures
# the standing 29.26 t/s figure under this env rather than quoting it, because
# that number was taken at GGML_CUDA_ALLREDUCE=internal, not none.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export SPLIT=tensor GGML_CUDA_ALLREDUCE=none \
       REPS=3 N_PREDICT=400 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt \
       BATCH=2048

for ub in 512 2048; do
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 70 ]] && break
        echo "cooling: hottest card ${t}C"; sleep 20
    done
    UB="$ub" ./scripts/run-spec-placement.sh "h24-ub${ub}-mtp-16k" draft-mtp \
        /root/mtp-Qwen3.8-27B-Q4_0.gguf default
    echo "--- ub=${ub} done (exit $?) ---"
    sleep 15   # let VRAM actually free before the next load
done

echo "=== H24 results ==="
grep -E '^(label|h24-)' results/h11-placement.csv | column -s, -t

git add -A
if git diff --cached --quiet; then
    echo "nothing to commit"
else
    git commit -q -m "bench: h24 ubatch-vs-decode at 16k (ub 512 vs 2048, tensor + MTP)"
    git push -q && echo "pushed" || echo "WARNING: push failed; commit is local" >&2
fi
