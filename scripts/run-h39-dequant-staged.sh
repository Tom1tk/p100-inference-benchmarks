#!/usr/bin/env bash
# H39 (fork plan P3 leftover): GGML_PASCAL_DEQUANT_STAGED=1, 256-thread CTAs + smem-staged 16-byte stores
# for the 32-thread dequant-to-F16 kernels (q4_K, iq4_xs, iq3_s, iq2/iq1). Bit-identical by construction.
# Micro (MUL_MAT 4096x14336, n=512): q4_K -6.4%, iq3_s -7.9%, iq4_xs -4.5% of the whole op;
# q5_K got +1.6% so 64-thread kernels are excluded. 519/519 contiguous MUL_MAT cases pass.
# 1) Identity: Q4_K_M 2-card, -c 2048 --chunks 2, KLD on vs off. Pass = |KLD| <= 1e-5, same top 100%.
# 2) 1-card IQ3_S 16k, drafter-free, REPS=2. Control banked: h34-1card-iq3 rep 2, 15.83 / 196.72.
#    Skip step 3 if 1-card prefill gains under 1%.
# 3) 2-card serve command 16k, REPS=2. Control banked: h36-duplex-16k, 40.37 / 365.3.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK=/root/llama.cpp-Pascal/build-cuda-p100/bin
Q4=/root/Qwen3.8-27B-UD-Q4_K_M.gguf
IQ3=/root/Qwen3.8-27B-UD-IQ3_S.gguf
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 \
       GGML_CUDA_PASCAL_GEMM_ALGO=1 GGML_PASCAL_DUPLEX_REDUCE=1
cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 60 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20
    done
}
ppl() {  # label staged extra-args...
    cool
    ABORT_TEMP=80 INTERVAL=2 ./scripts/gpu-monitor.sh > "logs/h39-$1-temps.log" 2>/dev/null &
    local mon=$!
    GGML_PASCAL_DEQUANT_STAGED=$2 "$FORK/llama-perplexity" -m "$Q4" -f prompts/h26-100k-prompt.txt -c 2048 --chunks 2 \
        -b 2048 -ub 2048 -ngl 99 -fa 1 -sm tensor "${@:3}" > "logs/h39-$1.log" 2>&1
    local rc=$?; kill $mon
    echo "--- $1 rc=$rc, peak $(tr '|' '\n' < "logs/h39-$1-temps.log" | awk -F', ' 'NF>=3{print $2}' | sort -n | tail -1)C, $(grep -m1 -E 'Final estimate|Mean    KLD' "logs/h39-$1.log") ---"
    return $rc
}
fin() { git add -A results logs scripts/run-h39-dequant-staged.sh
    git commit -q -m "bench: H39 staged dequant ($1)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }; }

BASE=/root/h28-kld/h39-base.kld; mkdir -p /root/h28-kld
ppl off 0 --kl-divergence-base "$BASE" || { rm -f "$BASE"; fin "perplexity failed"; exit 1; }
ppl staged 1 --kl-divergence-base "$BASE" --kl-divergence || { rm -f "$BASE"; fin "perplexity failed"; exit 1; }
rm -f "$BASE"
K=$(grep -m1 'Mean    KLD:' logs/h39-staged.log | awk '{print $3}')
T=$(grep -m1 'Same top p:' logs/h39-staged.log | awk '{print $4}')
echo "staged vs off: mean KLD $K, same top $T"
python3 -c "import sys; sys.exit(0 if abs(float('$K')) <= 1e-5 and float('$T') >= 100 else 1)" \
    || { echo "gate: identity FAIL"; fin "identity FAILED"; exit 1; }
echo "gate: identity PASS"

export GGML_PASCAL_DEQUANT_STAGED=1 N_PREDICT=400 BATCH=2048 UB=2048 BIN=$FORK/llama-server
cool
REPS=2 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt SPLIT=none MODEL="$IQ3" \
    ./scripts/run-spec-placement.sh h39-1card-iq3-staged none none default
P=$(awk -F, '$1=="h39-1card-iq3-staged" && $5==2 {print $7}' results/h11-placement.csv | tail -1)
echo "1-card rep2 prefill $P (control 196.72)"
if python3 -c "import sys; sys.exit(0 if float('${P:-0}') >= 196.72*1.01 else 1)"; then
    cool
    REPS=2 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt SPLIT=tensor \
        ./scripts/run-spec-placement.sh h39-staged-16k draft-mtp /root/mtp-Qwen3.8-27B-Q4_0.gguf default
else
    echo "gate: 1-card gain under 1%, 2-card arm skipped"
fi
grep -E '^(label|h34-1card-iq3|h36-duplex-16k|h39-)' results/h11-placement.csv | column -s, -t
fin "identity + 1-card 16k + 2-card 16k"
