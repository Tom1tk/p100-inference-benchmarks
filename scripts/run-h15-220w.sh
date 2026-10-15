#!/usr/bin/env bash
# H15 step 2 (user call 2026-10-15: cap raised to 220 W by the user; REPS=2, 16k only). Controls banked: p15-200w-16k, p15-175w-16k.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 GGML_CUDA_PASCAL_GEMM_ALGO=1 \
       GGML_PASCAL_DUPLEX_REDUCE=1 GGML_PASCAL_MMVH_Q6K_WIDE=1 GGML_PASCAL_MMVH_LOWBIT=1 GGML_PASCAL_FA_GEMM=1
export N_PREDICT=400 BATCH=2048 UB=2048 BIN=/root/llama.cpp-Pascal/build-cuda-p100/bin/llama-server
cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 55 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20
    done
}
arm() {  # label cap
    nvidia-smi -pl "$2" >/dev/null || { echo "ERROR: could not set ${2} W"; exit 1; }
    echo "cap: $(nvidia-smi --query-gpu=power.limit --format=csv,noheader | paste -sd' ')"
    cool
    REPS=2 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt SPLIT=tensor \
        ./scripts/run-spec-placement.sh "$1" draft-mtp /root/mtp-Qwen3.8-27B-Q4_0-head96k.gguf default
    echo "$1 peak power per card: $(tr '|' '\n' < "logs/$1.temps.log" | awk -F', ' 'NF>=3{c=$1; sub(/.* /,"",c); if ($3>m[c]) m[c]=$3} END{for (c in m) printf "gpu%s %.0f W  ", c, m[c]}')"
    git add -A results logs scripts/run-h15-220w.sh
    git commit -q -m "bench: H15 $1 (16k, REPS=2)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }
}
arm p15-220w-16k 220
python3 scripts/compare-outputs.py p15-200w-16k p15-220w-16k || true
grep -E '^(label|h45-head96k-16k|p15-)' results/h11-placement.csv | column -s, -t
