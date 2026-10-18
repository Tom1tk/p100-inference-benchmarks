#!/usr/bin/env bash
# H69: IQ4_XS on mmvh only at 1 column (GGML_PASCAL_MMVH_IQ4XS_1COL, default on sm_60); verify widths go to mmvq.
# 0) test-backend-ops MUL_MAT correctness for the UD types. 1) llama-bench pp4 (4-column verify) off vs on, -r 5;
# stop if pp4 gains < 3%. 2) KLD 8x512 at ub 4 (the verify width), off vs on, vs a fresh Q5_K_M FP32 reference;
# gate on <= off + 0.002. 3) 16k serve command REPS=3 vs h58-f16v-16k; pass >= +3% ms/target-step.
# 4) haystack (f16 KV, MTP), gate >= 18/19. 98k skipped (rule zero: 16k decides the lever).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unset $(env | grep -oE '^GGML_(CUDA_)?PASCAL_[A-Z0-9_]+' || true)
FORK=/root/llama.cpp-Pascal/build-cuda-p100/bin MTP=/root/mtp-Qwen3.8-27B-Q4_0-head96k.gguf
FP32=/root/llama.cpp-Pascal/build-h17/bin/llama-perplexity
Q5=/root/Qwen3.8-27B-UD-Q5_K_M.gguf UD=/root/Qwen3.8-27B-UD-Q4_K_M.gguf
BASE=/root/h28-kld/h69-base.kld CSV=results/h69-kld.csv
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader | paste -sd' ')
echo "cap: $cap"; [[ "$cap" == "200.00 W 200.00 W" ]] || { echo "ERROR: expected 200 W on both cards"; exit 1; }
cool() { for _ in $(seq 1 30); do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
    [[ "$t" -le 55 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20; done; }
fin() { git add -A results logs scripts/run-h69-iq4colerify.sh
    git commit -q -m "bench: H69 fp16 verify kernels ($1)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }; }
steps() { awk -F, -v l="$1" '$1==l && $16=="ok"{d+=$6; n++; s+=($8-$10); t+=$8/$6; a+=$11} END{if(n) printf "%s: decode %.2f, accept %.1f%%, %.2f ms/target-step (n=%d)\n", l, d/n, a/n, 1000*t/s, n}' results/h11-placement.csv; }

# 0) correctness (no model load)
CUDA_VISIBLE_DEVICES=0 $FORK/test-backend-ops test -o MUL_MAT -b CUDA0 -p 'type_a=iq4_xs' > logs/h69-tbo.log 2>&1
tail -3 logs/h69-tbo.log; grep -q "FAIL" logs/h69-tbo.log && { echo "STOP: test-backend-ops failures"; grep FAIL logs/h69-tbo.log | head; fin tbo-fail; exit 1; }

# 1) llama-bench
bench() { cool; ABORT_TEMP=80 INTERVAL=2 ./scripts/gpu-monitor.sh > "logs/h69-$1-temps.log" 2>/dev/null & local mon=$!
    env CUDA_VISIBLE_DEVICES=0,1 "${@:2}" $FORK/llama-bench -m $UD -sm tensor -ngl 99 -fa 1 -t 8 \
        -p 4 -n 32 -r 5 -o csv > "logs/h69-$1.csv" 2> "logs/h69-$1.log"; local rc=$?; kill $mon; return $rc; }
bench off GGML_PASCAL_MMVH_IQ4XS_1COL=0; bench on
python3 - <<'PY' > results/h69-bench.txt
import csv
g = {l: {('pp4' if r['n_gen']=='0' else 'tg32'): float(r['avg_ts']) for r in csv.DictReader(open(f'logs/h69-{l}.csv'))} for l in ('off','on')}
for t in ('pp4','tg32'): print(f"{t}: off {g['off'][t]:.2f} on {g['on'][t]:.2f} ({100*(g['on'][t]/g['off'][t]-1):+.1f}%)")
print('GO' if g['on']['pp4'] >= 1.03*g['off']['pp4'] else 'STOP')
PY
cat results/h69-bench.txt; fin bench
grep -q '^GO' results/h69-bench.txt || { echo "STOP: pp4 gain < 3%"; exit 0; }

# 2) KLD at the verify width
ppl() {  # label bin model ub extra...
    local label=$1 bin=$2 model=$3 ub=$4; shift 4; cool
    ABORT_TEMP=80 INTERVAL=2 ./scripts/gpu-monitor.sh > "logs/h69-$label-temps.log" 2>/dev/null & local mon=$!
    "$bin" -m "$model" -f prompts/h26-100k-prompt.txt -c 512 --chunks 8 -b 512 -ub "$ub" -ngl 99 -fa 1 -sm tensor "$@" \
        > "logs/h69-$label.log" 2>&1
    local rc=$?; kill $mon; echo "--- $label rc=$rc ---"; return $rc; }
arm() { ppl "$1" "$FORK/llama-perplexity" "$UD" 4 --kl-divergence-base "$BASE" --kl-divergence || return 1
    local f="logs/h69-$1.log"
    printf '%s,%s,%s,%s,%s\n' "$1" "$(grep -m1 'Mean    KLD:' "$f" | awk '{print $3}')" "$(grep -m1 '99.0%   KLD:' "$f" | awk '{print $3}')" \
        "$(grep -m1 '99.9%   KLD:' "$f" | awk '{print $3}')" "$(grep -m1 'Same top p:' "$f" | awk '{print $4}')" >> "$CSV"; }
echo "label,mean_kld,p99_kld,p999_kld,same_top_pct" > "$CSV"
GGML_CUDA_PASCAL_MMVH=0 ppl ref-q5-fp32 "$FP32" "$Q5" 512 --kl-divergence-base "$BASE" || { fin kld-ref-fail; exit 1; }
GGML_PASCAL_MMVH_IQ4XS_1COL=0 arm off; arm on
rm -f "$BASE"; column -s, -t "$CSV"; fin KLD
python3 - "$CSV" <<'PY' || { echo "STOP: KLD gate"; exit 0; }
import csv,sys; r={x['label']:float(x['mean_kld']) for x in csv.DictReader(open(sys.argv[1]))}
print(f"KLD off {r['off']:.4f} on {r['on']:.4f}"); sys.exit(0 if r['on'] <= r['off'] + 0.002 else 1)
PY

# 3) 16k serve command
export N_PREDICT=400 BATCH=2048 UB=2048 BIN=$FORK/llama-server REPS=3 SPLIT=tensor
cool; CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt ./scripts/run-spec-placement.sh h69-iq4col-16k draft-mtp "$MTP" default
steps h58-f16v-16k; steps h69-iq4col-16k; python3 scripts/compare-outputs.py h58-f16v-16k h69-iq4col-16k; fin 16k

# 4) haystack
cool; KV=f16 SPEC=mtp ./scripts/run-phase8-tier2.sh h69-iq4col-f16
hs=$(awk -F, '$1=="h69-iq4col-f16"{c+=$5} END{print c+0}' results/phase8-tier2.csv); echo "haystack: $hs/19 (h58 19/19)"; fin haystack
echo "=== H69 done ==="
