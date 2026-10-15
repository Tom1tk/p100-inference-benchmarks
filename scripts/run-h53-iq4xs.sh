#!/usr/bin/env bash
# H53: UD-IQ4_XS instead of UD-Q4_K_M on 2 cards, current serve env, 200 W cap.
# 1) KLD gate (H30 ladder style: 8x512 of the 100k prompt, ub 8 = decode numerics) against a fresh
#    Q5_K_M FP32 (build-h17) reference. Q4_K_M is re-measured in the same run under the current env.
#    Pass = IQ4_XS mean KLD <= Q4_K_M + 0.005 and same-top >= Q4_K_M - 0.5 pt. Fail stops the driver.
# 2) 16k decode REPS=3 vs the banked control p15-200w-16k(-r3) (f16 KV). Pass = >= 5%. Fail stops the driver.
# 3) 100k haystack, f16 KV. Pass = answers every question p8-ref (Q5_K_M reference) answers.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK=/root/llama.cpp-Pascal/build-cuda-p100/bin
FP32=/root/llama.cpp-Pascal/build-h17/bin/llama-perplexity
Q5=/root/Qwen3.8-27B-UD-Q5_K_M.gguf
Q4=/root/Qwen3.8-27B-UD-Q4_K_M.gguf
IQ4=/root/Qwen3.8-27B-UD-IQ4_XS.gguf
MTP=/root/mtp-Qwen3.8-27B-Q4_0-head96k.gguf
BASE=/root/h28-kld/h53-base.kld; mkdir -p /root/h28-kld
CSV=results/h53-kld.csv
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 GGML_CUDA_PASCAL_GEMM_ALGO=1 \
       GGML_PASCAL_DUPLEX_REDUCE=1 GGML_PASCAL_MMVH_Q6K_WIDE=1 GGML_PASCAL_MMVH_LOWBIT=1 GGML_PASCAL_FA_GEMM=1
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader | paste -sd' ')
echo "cap: $cap"; [[ "$cap" == "200.00 W 200.00 W" ]] || { echo "ERROR: expected 200 W on both cards"; exit 1; }
[[ -f "$IQ4" ]] || { echo "ERROR: $IQ4 missing"; exit 1; }
cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 55 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20
    done
}
fin() { git add -A results logs scripts/run-h53-iq4xs.sh scripts/run-phase8-tier2.sh
    git commit -q -m "bench: H53 UD-IQ4_XS ($1)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }; }
ppl() {  # label bin model ub extra-args...
    local label=$1 bin=$2 model=$3 ub=$4; shift 4
    cool
    ABORT_TEMP=80 INTERVAL=2 ./scripts/gpu-monitor.sh > "logs/h53-$label-temps.log" 2>/dev/null &
    local mon=$!
    "$bin" -m "$model" -f prompts/h26-100k-prompt.txt -c 512 --chunks 8 -b 512 -ub "$ub" -ngl 99 -fa 1 -sm tensor "$@" \
        > "logs/h53-$label.log" 2>&1
    local rc=$?; kill $mon
    echo "--- $label rc=$rc, peak $(tr '|' '\n' < "logs/h53-$label-temps.log" | awk -F', ' 'NF>=3{print $2}' | sort -n | tail -1)C ---"
    return $rc
}
arm() {  # label model
    ppl "$1" "$FORK/llama-perplexity" "$2" 8 --kl-divergence-base "$BASE" --kl-divergence || return 1
    local f="logs/h53-$1.log"
    printf '%s,%s,%s,%s,%s\n' "$1" \
        "$(grep -m1 'Mean    KLD:' "$f" | awk '{print $3}')" "$(grep -m1 '99.0%   KLD:' "$f" | awk '{print $3}')" \
        "$(grep -m1 '99.9%   KLD:' "$f" | awk '{print $3}')" "$(grep -m1 'Same top p:' "$f" | awk '{print $4}')" >> "$CSV"
}
mean() { awk -F, -v re="$1" '$1 ~ re && $16=="ok" {s+=$6; n++} END {if (n) printf "%.2f", s/n; else print 0}' results/h11-placement.csv; }

# --- 1) KLD gate ---
echo "label,mean_kld,p99_kld,p999_kld,same_top_pct" > "$CSV"
GGML_CUDA_PASCAL_MMVH=0 ppl ref-q5-fp32 "$FP32" "$Q5" 512 --kl-divergence-base "$BASE" || { rm -f "$BASE"; fin "reference failed"; exit 1; }
arm q4km "$Q4" && arm iq4xs "$IQ4"; rc=$?
rm -f "$BASE"
column -s, -t "$CSV"
[[ $rc -eq 0 ]] || { fin "KLD arm failed"; exit 1; }
read -r KQ TQ < <(awk -F, '$1=="q4km"{print $2, $5}' "$CSV")
read -r KI TI < <(awk -F, '$1=="iq4xs"{print $2, $5}' "$CSV")
echo "KLD: q4km $KQ (top $TQ) | iq4xs $KI (top $TI)"
python3 -c "import sys; sys.exit(0 if float('$KI') <= float('$KQ')+0.005 and float('$TI') >= float('$TQ')-0.5 else 1)" \
    || { echo "gate: KLD FAIL"; fin "KLD gate FAILED"; exit 1; }
echo "gate: KLD PASS"; fin "KLD gate"

# --- 2) 16k decode ---
cool
MODEL="$IQ4" N_PREDICT=400 BATCH=2048 UB=2048 BIN=$FORK/llama-server REPS=3 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt \
    SPLIT=tensor ./scripts/run-spec-placement.sh h53-iq4xs-16k draft-mtp "$MTP" default
C16=$(mean '^p15-200w-16k(-r3)?$'); I16=$(mean '^h53-iq4xs-16k$')
echo "16k decode: control $C16, iq4xs $I16"
fin "16k decode"
python3 -c "import sys; sys.exit(0 if float('$I16') >= 1.05*float('$C16') else 1)" \
    || { echo "skip: iq4xs not 5% over control, haystack cancelled"; exit 0; }

# --- 3) 100k haystack ---
cool
BIN=$FORK/llama-server MODEL="$IQ4" KV=f16 SPEC=mtp ./scripts/run-phase8-tier2.sh p8-h53-iq4xs
awk -F, '$1=="p8-ref"||$1=="p8-h53-iq4xs"{n[$1]++; h[$1]+=$5} END{for(l in n) print l, h[l]"/"n[l]}' results/phase8-tier2.csv
awk -F, '$1=="p8-ref"&&$5==1{r[$2","$3]=1} $1=="p8-h53-iq4xs"&&$5!=1&&r[$2","$3]{print "MISS vs ref:", $2, $3}' results/phase8-tier2.csv
cat logs/h53-*.cap 2>/dev/null
fin "100k haystack"
