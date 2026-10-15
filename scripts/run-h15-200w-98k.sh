#!/usr/bin/env bash
# H15 step 4 (user call 2026-10-15: cap set to 200 W by user, "98k 2x run"). Cap is NOT changed here.
# Control banked at 175 W: h46-fagemm-100k + -r23 (REPS=3, 275.75 / 28.38).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 GGML_CUDA_PASCAL_GEMM_ALGO=1 \
       GGML_PASCAL_DUPLEX_REDUCE=1 GGML_PASCAL_MMVH_Q6K_WIDE=1 GGML_PASCAL_MMVH_LOWBIT=1 GGML_PASCAL_FA_GEMM=1
export N_PREDICT=400 BATCH=2048 UB=2048 BIN=/root/llama.cpp-Pascal/build-cuda-p100/bin/llama-server
L=p15-200w-98k
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader | paste -sd' ')
echo "cap: $cap"; [[ "$cap" == "200.00 W 200.00 W" ]] || { echo "ERROR: expected 200 W on both cards"; exit 1; }
for _ in $(seq 1 30); do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
    [[ "$t" -le 55 ]] && break; echo "cooling: hottest card ${t}C"; sleep 20
done
REPS=2 CTX=100000 CTK=q8_0 CTV=q8_0 PROMPT_FILE=prompts/h26-100k-prompt.txt NP=1 REQ_TIMEOUT=1800 SPLIT=tensor \
    ./scripts/run-spec-placement.sh "$L" draft-mtp /root/mtp-Qwen3.8-27B-Q4_0-head96k.gguf default
echo "$L peak power per card: $(tr '|' '\n' < "logs/$L.temps.log" | awk -F', ' 'NF>=3{c=$1; sub(/.* /,"",c); if ($3>m[c]) m[c]=$3} END{for (c in m) printf "gpu%s %.0f W  ", c, m[c]}')"
python3 scripts/compare-outputs.py h46-fagemm-100k "$L" || true
grep -E "^(label|h46-fagemm-100k|$L)" results/h11-placement.csv | column -s, -t
git add -A results logs scripts/run-h15-200w-98k.sh
git commit -q -m "bench: H15 $L (98k, REPS=2, 200 W)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }
