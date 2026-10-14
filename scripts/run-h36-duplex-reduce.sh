#!/usr/bin/env bash
# H36: GGML_PASCAL_DUPLEX_REDUCE=1 lets the two directions of each 2-card AllReduce copy overlap
# (P2 step 0: they run back to back, 0->1 then 1->0, ~2.06 ms each, ~260 ms per pp2048 = 5.4%).
# Ordering-only change, so numerics must be identical.
# 1) Identity: H35's q4-fp16-algo3 perplexity config (ub 512, 8 chunks), flag off saves the logits,
#    flag on is scored against them. Pass = mean KLD <= 1e-5 and same-top 100% (H34 f16 reduce: -0.000005).
# 2) Speed: serve-fork-16k (39.48 / 353.05 banked today) + DUPLEX=1, REPS=2.
# 1-card arm skipped (rule zero): the flag only touches peer copies; one card makes none.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK=/root/llama.cpp-Pascal/build-cuda-p100/bin
Q4=/root/Qwen3.8-27B-UD-Q4_K_M.gguf
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1
cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 60 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20
    done
}
ppl() {  # label duplex extra-args...
    cool
    ABORT_TEMP=80 INTERVAL=2 ./scripts/gpu-monitor.sh > "logs/h36-$1-temps.log" 2>/dev/null &
    local mon=$!
    GGML_PASCAL_DUPLEX_REDUCE=$2 GGML_CUDA_PASCAL_GEMM_ALGO=1 "$FORK/llama-perplexity" -m "$Q4" \
        -f prompts/h26-100k-prompt.txt -c 512 --chunks 8 -b 512 -ub 512 -ngl 99 -fa 1 -sm tensor \
        "${@:3}" > "logs/h36-$1.log" 2>&1
    local rc=$?; kill $mon
    echo "--- $1 rc=$rc, peak $(tr '|' '\n' < "logs/h36-$1-temps.log" | awk -F', ' 'NF>=3{print $2}' | sort -n | tail -1)C, $(grep -m1 -E 'Final estimate|Mean    KLD' "logs/h36-$1.log") ---"
    return $rc
}
fin() { git add -A results logs scripts/run-h36-duplex-reduce.sh
    git commit -q -m "bench: H36 duplex AllReduce copies ($1)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }; }

BASE=/root/h28-kld/h36-base.kld; mkdir -p /root/h28-kld
ppl duplex0 0 --kl-divergence-base "$BASE" || { fin "perplexity failed"; exit 1; }
ppl duplex1 1 --kl-divergence-base "$BASE" --kl-divergence || { fin "perplexity failed"; exit 1; }
rm -f "$BASE"
K=$(grep -m1 'Mean    KLD:' logs/h36-duplex1.log | awk '{print $3}')
T=$(grep -m1 'Same top p:' logs/h36-duplex1.log | awk '{print $4}')
echo "duplex1 vs off: mean KLD $K, same top $T"
python3 -c "import sys; sys.exit(0 if abs(float('$K')) <= 1e-5 and float('$T') >= 100 else 1)" \
    || { echo "gate: identity FAIL"; fin "identity FAILED"; exit 1; }
echo "gate: identity PASS"

export GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 GGML_CUDA_PASCAL_GEMM_ALGO=1 GGML_PASCAL_DUPLEX_REDUCE=1 \
       REPS=2 N_PREDICT=400 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt \
       BATCH=2048 UB=2048 BIN=$FORK/llama-server
cool
SPLIT=tensor ./scripts/run-spec-placement.sh h36-duplex-16k draft-mtp /root/mtp-Qwen3.8-27B-Q4_0.gguf default
grep -E '^(label|serve-fork-16k|h36-)' results/h11-placement.csv | column -s, -t
fin "identity + 16k speed"
