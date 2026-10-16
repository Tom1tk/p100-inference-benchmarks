#!/usr/bin/env bash
# H56: fork tidy-up + rebase onto mainline 7182ef7 (refuted levers removed, adopted levers default-on for sm_60).
# Runs with NO Pascal env vars, only the two upstream ones, so it also proves the defaults switch on.
# 2-card 16k (serve command), REPS=3, vs r10s-fork-16k (47.54 / 392.1). Then 1-card IQ3_S 16k + staged dequant,
# REPS=3, vs h39r-1card-staged (18.39 / 220.34). Pass: decode and prefill >= 98% of control, output identical.
# A 2-card fail stops the driver before the 1-card arm.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unset $(env | grep -oE '^GGML_(CUDA_)?PASCAL_[A-Z0-9_]+' || true)
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1
export N_PREDICT=400 BATCH=2048 UB=2048 BIN=/root/llama.cpp-Pascal/build-cuda-p100/bin/llama-server
MTP=/root/mtp-Qwen3.8-27B-Q4_0-head96k.gguf
IQ3=/root/Qwen3.8-27B-UD-IQ3_S.gguf
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader | paste -sd' ')
echo "cap: $cap"; [[ "$cap" == "200.00 W 200.00 W" ]] || { echo "ERROR: expected 200 W on both cards"; exit 1; }
env | grep -E '^GGML_' | sort
cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 55 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20
    done
}
fin() { git add -A results logs scripts/run-h56-rebase-check.sh
    git commit -q -m "bench: H56 rebase check ($1)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }; }
mean() { awk -F, -v l="$1" -v c="$2" '$1==l && $16=="ok" {s+=$c; n++} END {if (n) printf "%.2f", s/n; else print 0}' results/h11-placement.csv; }
check() {  # label control
    local d p cd cp same
    d=$(mean "$1" 6); p=$(mean "$1" 7); cd=$(mean "$2" 6); cp=$(mean "$2" 7)
    same=$(python3 scripts/compare-outputs.py "$2" "$1"); echo "$same"
    echo "$1: decode $d vs $cd, prefill $p vs $cp"
    [[ "$same" == *identical* ]] && python3 -c "import sys; sys.exit(0 if $d >= 0.98*$cd and $p >= 0.98*$cp else 1)"
}

cool
REPS=3 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt SPLIT=tensor \
    ./scripts/run-spec-placement.sh h56-rebase-16k draft-mtp "$MTP" default
fin h56-rebase-16k
check h56-rebase-16k r10s-fork-16k || { echo "FAIL: 2-card check, 1-card arm cancelled"; cat logs/h56-*.cap 2>/dev/null; fin "2-card FAIL"; exit 1; }
echo "PASS: 2-card"

cool
GGML_PASCAL_DEQUANT_STAGED=1 REPS=3 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt SPLIT=none MODEL="$IQ3" \
    ./scripts/run-spec-placement.sh h56-rebase-1card none none default
check h56-rebase-1card h39r-1card-staged && echo "PASS: 1-card" || echo "FAIL: 1-card"
grep -E '^h56-' results/h11-placement.csv | cut -d, -f1,5-7,16 | column -s, -t
cat logs/h56-*.cap 2>/dev/null
fin "summary"
