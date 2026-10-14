#!/usr/bin/env bash
# H41 (fork plan D2 follow-up): GGML_PASCAL_MMVH_Q6K_WIDE=1, Q6_K mmvh with 4 lanes per superblock.
# 1) Decode-path KLD: Q4_K_M 2-card, -c 2048 --chunks 2 -ub 8, wide vs off. Pass = |KLD| <= 1e-4, same top >= 99.9%.
# 2) 2-card serve command 16k, REPS=2. Control banked: h36-duplex-16k, 40.37 / 365.3.
#    No 1-card arm: UD-IQ3_S has ~3% Q6_K bytes, expected <= 1% (rule zero, see HYPOTHESES H41).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK=/root/llama.cpp-Pascal/build-cuda-p100/bin
Q4=/root/Qwen3.8-27B-UD-Q4_K_M.gguf
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 \
       GGML_CUDA_PASCAL_GEMM_ALGO=1 GGML_PASCAL_DUPLEX_REDUCE=1
cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 60 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20
    done
}
ppl() {  # label wide extra-args...
    cool
    ABORT_TEMP=80 INTERVAL=2 ./scripts/gpu-monitor.sh > "logs/h41-$1-temps.log" 2>/dev/null &
    local mon=$!
    GGML_PASCAL_MMVH_Q6K_WIDE=$2 "$FORK/llama-perplexity" -m "$Q4" -f prompts/h26-100k-prompt.txt -c 2048 --chunks 2 \
        -b 2048 -ub 8 -ngl 99 -fa 1 -sm tensor "${@:3}" > "logs/h41-$1.log" 2>&1
    local rc=$?; kill $mon
    echo "--- $1 rc=$rc, peak $(tr '|' '\n' < "logs/h41-$1-temps.log" | awk -F', ' 'NF>=3{print $2}' | sort -n | tail -1)C, $(grep -m1 -E 'Final estimate|Mean    KLD' "logs/h41-$1.log") ---"
    return $rc
}
fin() { git add -A results logs scripts/run-h41-q6k-wide.sh HYPOTHESES.md
    git commit -q -m "bench: H41 Q6_K wide mmvh ($1)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }; }

BASE=/root/h28-kld/h41-base.kld; mkdir -p /root/h28-kld
ppl off 0 --kl-divergence-base "$BASE" || { rm -f "$BASE"; fin "perplexity failed"; exit 1; }
ppl wide 1 --kl-divergence-base "$BASE" --kl-divergence || { rm -f "$BASE"; fin "perplexity failed"; exit 1; }
rm -f "$BASE"
K=$(grep -m1 'Mean    KLD:' logs/h41-wide.log | awk '{print $3}')
T=$(grep -m1 'Same top p:' logs/h41-wide.log | awk '{print $4}')
echo "wide vs off: mean KLD $K, same top $T"
python3 -c "import sys; sys.exit(0 if abs(float('$K')) <= 1e-4 and float('$T') >= 99.9 else 1)" \
    || { echo "gate: KLD FAIL"; fin "KLD FAILED"; exit 1; }
echo "gate: KLD PASS"

export GGML_PASCAL_MMVH_Q6K_WIDE=1 N_PREDICT=400 BATCH=2048 UB=2048 BIN=$FORK/llama-server
cool
REPS=2 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt SPLIT=tensor \
    ./scripts/run-spec-placement.sh h41-wide-16k draft-mtp /root/mtp-Qwen3.8-27B-Q4_0.gguf default
python3 scripts/compare-outputs.py h36-duplex-16k h41-wide-16k || true  # fp32 reorder: may differ
grep -E '^(label|h36-duplex-16k|h41-)' results/h11-placement.csv | column -s, -t
fin "KLD + 2-card 16k"
