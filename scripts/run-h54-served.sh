#!/usr/bin/env bash
# H54 served: GGML_PASCAL_FA_GQA6=1 (fork 7b6a784: tile FA packs all 6 Q heads per K/V head for nb <= 4, D256).
# 1) 2-card 98k decode, CACHE_PROMPT=1, REPS=3, n=3, f16 KV vs banked h49-n3-98k (same path and build bar
#    GQA6/R4; R4 is inert at <= 4 columns). Pass: decode >= 5% over control.
# 2) 1-card arm: IQ3_S 16k -sm none (drafter-free, nb 1, also GQA 6), staged env, REPS=3 vs banked h39r-1card-staged.
# 3) Only if 1) passed and the 98k output differs from the control: 100k haystack, f16 KV (pass = every question p8-ref answers).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 GGML_CUDA_PASCAL_GEMM_ALGO=1 \
       GGML_PASCAL_DUPLEX_REDUCE=1 GGML_PASCAL_MMVH_Q6K_WIDE=1 GGML_PASCAL_MMVH_LOWBIT=1 GGML_PASCAL_FA_GEMM=1 GGML_PASCAL_FA_GQA6=1
export N_PREDICT=400 BATCH=2048 UB=2048 BIN=/root/llama.cpp-Pascal/build-cuda-p100/bin/llama-server
MTP=/root/mtp-Qwen3.8-27B-Q4_0-head96k.gguf
IQ3=/root/Qwen3.8-27B-UD-IQ3_S.gguf
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader | paste -sd' ')
echo "cap: $cap"; [[ "$cap" == "200.00 W 200.00 W" ]] || { echo "ERROR: expected 200 W on both cards"; exit 1; }
cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 55 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20
    done
}
fin() { git add -A results logs scripts/run-h54-served.sh scripts/run-phase8-tier2.sh
    git commit -q -m "bench: H54 FA GQA6 served ($1)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }; }
mean() { awk -F, -v re="$1" -v c="$2" '$1 ~ re && $16=="ok" {s+=$c; n++} END {if (n) printf "%.2f", s/n; else print 0}' results/h11-placement.csv; }

# --- 1) 98k ---
cool
CACHE_PROMPT=1 REPS=3 CTX=100000 CTK=f16 CTV=f16 PROMPT_FILE=prompts/h26-100k-prompt.txt NP=1 REQ_TIMEOUT=1800 SPLIT=tensor \
    ./scripts/run-spec-placement.sh h54-gqa6-98k draft-mtp "$MTP" default
python3 scripts/compare-outputs.py h49-n3-98k h54-gqa6-98k; SAME=$?
C=$(mean '^h49-n3-98k$' 6); G=$(mean '^h54-gqa6-98k$' 6)
echo "98k decode: control h49-n3-98k $C, gqa6 $G ($(python3 -c "print(f'{100*($G/$C-1):+.1f}%')"))"
fin "h54-gqa6-98k"
PASS=0; python3 -c "import sys; sys.exit(0 if float('$G') >= 1.05*float('$C') else 1)" && PASS=1

# --- 2) 1-card ---
cool
GGML_PASCAL_DEQUANT_STAGED=1 REPS=3 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt SPLIT=none MODEL="$IQ3" \
    ./scripts/run-spec-placement.sh h54-1card-gqa6 none none default
python3 scripts/compare-outputs.py h39r-1card-staged h54-1card-gqa6 || echo "WARNING: 1-card output differs"
echo "1-card 16k decode: control $(mean '^h39r-1card-staged$' 6), gqa6 $(mean '^h54-1card-gqa6$' 6)"
fin "h54-1card-gqa6"

# --- 3) haystack ---
if [[ $PASS -eq 1 && $SAME -ne 0 ]]; then
    cool
    KV=f16 SPEC=mtp ./scripts/run-phase8-tier2.sh p8-h54-gqa6
    awk -F, '$1=="p8-ref"||$1=="p8-h54-gqa6"{n[$1]++; h[$1]+=$5} END{for(l in n) print l, h[l]"/"n[l]}' results/phase8-tier2.csv
    awk -F, '$1=="p8-ref"&&$5==1{r[$2","$3]=1} $1=="p8-h54-gqa6"&&$5!=1&&r[$2","$3]{print "MISS vs ref:", $2, $3}' results/phase8-tier2.csv
    fin "haystack"
else
    echo "skip haystack: pass=$PASS same_output=$((SAME==0))"
fi
grep -E '^(h49-n3-98k|h54-|h39r-1card-staged)' results/h11-placement.csv | cut -d, -f1,5-8,11,16 | column -s, -t
cat logs/h54-*.cap 2>/dev/null
fin "summary"
