#!/usr/bin/env bash
# H62: standard Q4_K_M (lmstudio, Q4_K 73% + Q6_K 27% by bytes) vs UD-Q4_K_M (7 weight types), rebased fork,
# serve command, 200 W. 1) KLD 8x512 ub 8 vs a fresh Q5_K_M FP32 reference, UD re-measured in the same run
# (gate: lms <= UD + 0.005, same-top >= UD - 0.5 pt; informative here, user asked for the speed arms regardless).
# 2) 16k REPS=3 vs h56-rebase-16k. 3) 98k REPS=3 vs r10s-fork-100k (pre-rebase; compare ms per target step).
# Pass: decode >= +3%. Haystack only if it passes.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unset $(env | grep -oE '^GGML_(CUDA_)?PASCAL_[A-Z0-9_]+' || true)
FORK=/root/llama.cpp-Pascal/build-cuda-p100/bin
FP32=/root/llama.cpp-Pascal/build-h17/bin/llama-perplexity
Q5=/root/Qwen3.8-27B-UD-Q5_K_M.gguf UD=/root/Qwen3.8-27B-UD-Q4_K_M.gguf LMS=/root/Qwen3.8-27B-Q4_K_M-lms.gguf
MTP=/root/mtp-Qwen3.8-27B-Q4_0-head96k.gguf
BASE=/root/h28-kld/h62-base.kld CSV=results/h62-kld.csv
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader | paste -sd' ')
echo "cap: $cap"; [[ "$cap" == "200.00 W 200.00 W" ]] || { echo "ERROR: expected 200 W on both cards"; exit 1; }
cool() { for _ in $(seq 1 30); do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
    [[ "$t" -le 55 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20; done; }
fin() { git add -A results logs scripts/run-h62-q4km.sh
    git commit -q -m "bench: H62 standard Q4_K_M ($1)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }; }
ppl() {  # label bin model ub extra...
    local label=$1 bin=$2 model=$3 ub=$4; shift 4; cool
    ABORT_TEMP=80 INTERVAL=2 ./scripts/gpu-monitor.sh > "logs/h62-$label-temps.log" 2>/dev/null &
    local mon=$!
    "$bin" -m "$model" -f prompts/h26-100k-prompt.txt -c 512 --chunks 8 -b 512 -ub "$ub" -ngl 99 -fa 1 -sm tensor "$@" \
        > "logs/h62-$label.log" 2>&1
    local rc=$?; kill $mon; echo "--- $label rc=$rc ---"; return $rc; }
arm() { ppl "$1" "$FORK/llama-perplexity" "$2" 8 --kl-divergence-base "$BASE" --kl-divergence || return 1
    local f="logs/h62-$1.log"
    printf '%s,%s,%s,%s,%s\n' "$1" "$(grep -m1 'Mean    KLD:' "$f" | awk '{print $3}')" "$(grep -m1 '99.0%   KLD:' "$f" | awk '{print $3}')" \
        "$(grep -m1 '99.9%   KLD:' "$f" | awk '{print $3}')" "$(grep -m1 'Same top p:' "$f" | awk '{print $4}')" >> "$CSV"; }
steps() {  # label -> mean decode, accept, ms per target step
    awk -F, -v l="$1" '$1==l && $16=="ok"{d+=$6; n++; s+=($8-$10); t+=$8/$6; a+=$11} END{if(n) printf "%s: decode %.2f, accept %.1f%%, %.2f ms/target-step (n=%d)\n", l, d/n, a/n, 1000*t/s, n}' results/h11-placement.csv; }

mkdir -p /root/h28-kld; echo "label,mean_kld,p99_kld,p999_kld,same_top_pct" > "$CSV"
GGML_CUDA_PASCAL_MMVH=0 ppl ref-q5-fp32 "$FP32" "$Q5" 512 --kl-divergence-base "$BASE" && { arm ud "$UD"; arm lms "$LMS"; }
rm -f "$BASE"; column -s, -t "$CSV"; fin "KLD"

export N_PREDICT=400 BATCH=2048 UB=2048 BIN=$FORK/llama-server REPS=3 SPLIT=tensor MODEL=$LMS
cool; CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt ./scripts/run-spec-placement.sh h62-lms-16k draft-mtp "$MTP" default
steps h56-rebase-16k; steps h62-lms-16k; fin 16k
cool; CTX=100000 NP=1 REQ_TIMEOUT=1800 PROMPT_FILE=prompts/h26-100k-prompt.txt ./scripts/run-spec-placement.sh h62-lms-98k draft-mtp "$MTP" default
steps r10s-fork-100k; steps h62-lms-98k
grep -E '^(h62-lms|h56-rebase-16k|r10s-fork-100k),' results/h11-placement.csv | cut -d, -f1,5-7,11-13,16 | column -s, -t
fin 98k
echo "=== H62 done ==="
