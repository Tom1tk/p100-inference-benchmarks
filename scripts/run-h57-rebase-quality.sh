#!/usr/bin/env bash
# H57: quality gate for the rebased fork (7182ef7 + tidy). 100k haystack, 19 questions, serve config
# (f16 KV, MTP), no Pascal env vars (sm_60 defaults). Pass: 19/19 as p8-final-f16 (old base); the Q5 FP32
# reference p8-ref gets 18/19. Then the H56 1-card arm the 2-card fail cancelled: IQ3_S 16k + staged dequant,
# REPS=3 vs h39r-1card-staged (18.39 / 220.34).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unset $(env | grep -oE '^GGML_(CUDA_)?PASCAL_[A-Z0-9_]+' || true)
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 BIN=/root/llama.cpp-Pascal/build-cuda-p100/bin/llama-server
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader | paste -sd' ')
echo "cap: $cap"; [[ "$cap" == "200.00 W 200.00 W" ]] || { echo "ERROR: expected 200 W on both cards"; exit 1; }
env | grep -E '^GGML_' | sort
cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 55 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20
    done
}
fin() { git add -A results logs scripts/run-h57-rebase-quality.sh
    git commit -q -m "bench: H57 rebase quality ($1)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }; }

cool
KV=f16 SPEC=mtp ./scripts/run-phase8-tier2.sh h57-rebase-f16
echo "haystack: $(awk -F, '$1=="h57-rebase-f16"{n++; c+=$5} END{print c "/" n}' results/phase8-tier2.csv) (p8-final-f16 19/19, p8-ref 18/19)"
fin haystack

cool
GGML_PASCAL_DEQUANT_STAGED=1 N_PREDICT=400 BATCH=2048 UB=2048 REPS=3 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt \
    SPLIT=none MODEL=/root/Qwen3.8-27B-UD-IQ3_S.gguf ./scripts/run-spec-placement.sh h57-rebase-1card none none default
python3 scripts/compare-outputs.py h39r-1card-staged h57-rebase-1card
grep -E '^(h57-|h39r-1card-staged,)' results/h11-placement.csv | cut -d, -f1,5-7,16 | column -s, -t
fin 1card
