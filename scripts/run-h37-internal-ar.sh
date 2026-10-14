#!/usr/bin/env bash
# H37 (fork plan D3): one-shot small-message AllReduce. Upstream's GGML_CUDA_ALLREDUCE=internal chunked
# kernel (one launch per reduce, in-kernel cross-GPU sync on pinned host flags), ported to sm60, capped
# by GGML_PASCAL_AR_INTERNAL_MAX_BYTES so only decode/verify-sized reduces (< 1 MB, = upstream's
# copy-engine threshold) go internal; prefill keeps the meta butterfly (P2P + F16 + duplex).
# Decode trace: ~124 reduces/token with ~100 us matvec->add gaps (~12 ms of 53.6 ms/token).
# BF16 wire forced off (GGML_CUDA_AR_BF16_THRESHOLD=0): f32, a two-term sum, so numerics should match.
# 1) Identity at -ub 16 (reduces ~327 KB, exercises the chunked kernel): internal vs none,
#    pass = mean KLD <= 1e-5 and same-top 100%.
# 2) Speed: h36-duplex-16k (40.37 / 365.3) + internal. Prediction: decode up, prefill flat.
# 1-card arm skipped (rule zero): one card makes no reduces.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK=/root/llama.cpp-Pascal/build-cuda-p100/bin
Q4=/root/Qwen3.8-27B-UD-Q4_K_M.gguf
export GGML_CUDA_P2P=1 GGML_PASCAL_AR_INTERNAL_MAX_BYTES=1048575 GGML_CUDA_AR_BF16_THRESHOLD=0
cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 60 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20
    done
}
ppl() {  # label allreduce-mode extra-args...
    cool
    ABORT_TEMP=80 INTERVAL=2 ./scripts/gpu-monitor.sh > "logs/h37-$1-temps.log" 2>/dev/null &
    local mon=$!
    GGML_CUDA_ALLREDUCE=$2 GGML_CUDA_PASCAL_GEMM_ALGO=1 "$FORK/llama-perplexity" -m "$Q4" \
        -f prompts/h26-100k-prompt.txt -c 512 --chunks 8 -b 512 -ub 16 -ngl 99 -fa 1 -sm tensor \
        "${@:3}" > "logs/h37-$1.log" 2>&1
    local rc=$?; kill $mon
    echo "--- $1 rc=$rc, peak $(tr '|' '\n' < "logs/h37-$1-temps.log" | awk -F', ' 'NF>=3{print $2}' | sort -n | tail -1)C, $(grep -m1 -E 'Final estimate|Mean    KLD' "logs/h37-$1.log") ---"
    return $rc
}
fin() { git add -A results logs scripts/run-h37-internal-ar.sh
    git commit -q -m "bench: H37 internal small-message AllReduce ($1)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }; }

BASE=/root/h28-kld/h37-base.kld; mkdir -p /root/h28-kld
ppl none none --kl-divergence-base "$BASE" || { fin "perplexity failed"; exit 1; }
ppl internal internal --kl-divergence-base "$BASE" --kl-divergence || { fin "perplexity failed"; exit 1; }
rm -f "$BASE"
K=$(grep -m1 'Mean    KLD:' logs/h37-internal.log | awk '{print $3}')
T=$(grep -m1 'Same top p:' logs/h37-internal.log | awk '{print $4}')
echo "internal vs none: mean KLD $K, same top $T"
python3 -c "import sys; sys.exit(0 if abs(float('$K')) <= 1e-5 and float('$T') >= 100 else 1)" \
    || { echo "gate: identity FAIL"; fin "identity FAILED"; exit 1; }
echo "gate: identity PASS"

export GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 GGML_CUDA_PASCAL_GEMM_ALGO=1 GGML_PASCAL_DUPLEX_REDUCE=1 GGML_CUDA_ALLREDUCE=internal \
       REPS=2 N_PREDICT=400 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt \
       BATCH=2048 UB=2048 BIN=$FORK/llama-server
cool
SPLIT=tensor ./scripts/run-spec-placement.sh h37-internal-16k draft-mtp /root/mtp-Qwen3.8-27B-Q4_0.gguf default
grep -E '^(label|h36-duplex-16k|h37-)' results/h11-placement.csv | column -s, -t
fin "identity + 16k speed"
