#!/usr/bin/env bash
# Does TurboQuant KV pay enough to cover buun's 12-14% prefill deficit?
#
# buun -sm tensor matches rebased mainline on decode (+1.6%) but loses 12-14%
# on prefill (193.50/191.86/188.57/182.75 vs 222.75/223.43/214.88/208.25).
# The only reason to accept that is TurboQuant, which buun alone has. So the
# question is what turbo KV buys at depth, where KV traffic dominates decode.
#
# Also settles H7 for free: (turbo3, f16) is claimed to abort at fattn.cu:348.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

while pgrep -f 'run-best-config.sh|llama-server|llama-bench' >/dev/null; do sleep 30; done

export BIN=/root/buun-llama-cpp/build-cuda-p100/bin/llama-server
export SPLIT=tensor GGML_CUDA_ALLREDUCE=none REPS=3 N_PREDICT=400 \
       CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt

cool() {
    for _ in $(seq 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 70 ]] && return 0
        echo "cooling: hottest ${t}C"; sleep 20
    done
}
run() {
    local label="$1" ctk="$2" ctv="$3"
    cool
    echo "########## $label (ctk=$ctk ctv=$ctv)"
    CTK="$ctk" CTV="$ctv" ./scripts/run-spec-placement.sh \
        "$label" draft-mtp /root/mtp-Qwen3.8-27B-Q4_0.gguf default 2>&1 | tail -8
}

# Baseline on buun, directly comparable to p2-tensor-mtp-16k (29.26 on rebased).
run p3-buun-mtp-16k-f16       f16    f16
# H7 check -- expected to abort. REPS=1 so a failure costs nothing.
REPS=1 run p3-buun-mtp-16k-h7 turbo3 f16
# The combinations H7 says are legal.
run p3-buun-mtp-16k-t3t3      turbo3 turbo3
run p3-buun-mtp-16k-t3q8      turbo3 q8_0

echo "=== buun turbo sweep complete ==="
