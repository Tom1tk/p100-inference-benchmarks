#!/usr/bin/env bash
# H39r: re-test of H39 GGML_PASCAL_DEQUANT_STAGED=1 (bit-identical) on the 1-card arm at the 200 W running cap.
# 2-card re-tests are parked: H52 telemetry shows 0 s power-capped at 200 W on 2 cards.
# 1-card IQ3_S 16k, drafter-free, current env (as h46-1card-iq3-fagemm), REPS=3 each:
#   fresh control at 200 W (banked 1-card rows are 175 W), then staged.
# Pass: prefill >= 2% over the control mean, output identical. Cap counters logged per arm (H52, 1 card).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 GGML_CUDA_PASCAL_GEMM_ALGO=1 \
       GGML_PASCAL_DUPLEX_REDUCE=1 GGML_PASCAL_MMVH_Q6K_WIDE=1 GGML_PASCAL_MMVH_LOWBIT=1 GGML_PASCAL_FA_GEMM=1
export N_PREDICT=400 BATCH=2048 UB=2048 BIN=/root/llama.cpp-Pascal/build-cuda-p100/bin/llama-server
IQ3=/root/Qwen3.8-27B-UD-IQ3_S.gguf
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader | paste -sd' ')
echo "cap: $cap"; [[ "$cap" == "200.00 W 200.00 W" ]] || { echo "ERROR: expected 200 W on both cards"; exit 1; }
cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 55 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20
    done
}
fin() { git add -A results logs scripts/run-h39r-1card-200w.sh
    git commit -q -m "bench: H39r staged dequant 1-card at 200 W ($1)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }; }
pmean() { awk -F, -v l="$1" '$1==l && $16=="ok" {s+=$7; n++} END {if (n) printf "%.2f", s/n; else print 0}' results/h11-placement.csv; }
arm() {  # label staged
    cool
    GGML_PASCAL_DEQUANT_STAGED=$2 REPS=3 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt SPLIT=none MODEL="$IQ3" \
        ./scripts/run-spec-placement.sh "$1" none none default
    fin "$1"
}

arm h39r-1card-ctrl 0
arm h39r-1card-staged 1
python3 scripts/compare-outputs.py h39r-1card-ctrl h39r-1card-staged || echo "WARNING: output differs"
C=$(pmean h39r-1card-ctrl); S=$(pmean h39r-1card-staged)
echo "1-card 16k prefill: control $C, staged $S ($(python3 -c "print(f'{100*($S/$C-1):+.1f}%')"))"
grep -E '^h39r-' results/h11-placement.csv | cut -d, -f1,5-7,16 | column -s, -t
cat logs/h39r-*.cap
fin "summary"
