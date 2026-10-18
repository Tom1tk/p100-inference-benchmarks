#!/usr/bin/env bash
# H71: staged one-column Q6_K matvec (GGML_PASCAL_MMVQ_STAGE, default on sm_60), replaces mmvh for Q6_K n=1.
# 1) llama-bench tg32 + pp4, lever 0 vs default, -r 5. Stop if tg32 gains < 1.5%.
# 2) 16k serve command REPS=3 vs h58-f16v-16k, output must be byte-identical. 3) haystack only if it is not.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unset $(env | grep -oE '^GGML_(CUDA_)?PASCAL_[A-Z0-9_]+' || true)
FORK=/root/llama.cpp-Pascal/build-cuda-p100/bin MTP=/root/mtp-Qwen3.8-27B-Q4_0-head96k.gguf
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader | paste -sd' ')
echo "cap: $cap"; [[ "$cap" == "200.00 W 200.00 W" ]] || { echo "ERROR: expected 200 W on both cards"; exit 1; }
cool() { for _ in $(seq 1 30); do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
    [[ "$t" -le 55 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20; done; }
fin() { git add -A results logs scripts/run-h71b-q6stage.sh
    git commit -q -m "bench: H71b Q6_K staged matvec ($1)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }; }
steps() { awk -F, -v l="$1" '$1==l && $16=="ok"{d+=$6; n++; s+=($8-$10); t+=$8/$6; a+=$11} END{if(n) printf "%s: decode %.2f, accept %.1f%%, %.2f ms/target-step (n=%d)\n", l, d/n, a/n, 1000*t/s, n}' results/h11-placement.csv; }
bench() { cool; ABORT_TEMP=80 INTERVAL=2 ./scripts/gpu-monitor.sh > "logs/h71b-$1-temps.log" 2>/dev/null & local mon=$!
    env CUDA_VISIBLE_DEVICES=0,1 "${@:2}" $FORK/llama-bench -m /root/Qwen3.8-27B-UD-Q4_K_M.gguf -sm tensor -ngl 99 -fa 1 -t 8 \
        -p 4 -n 32 -r 5 -o csv > "logs/h71b-$1.csv" 2> "logs/h71b-$1.log"; local rc=$?; kill $mon
    python3 - "$1" <<'PY'
import csv,sys; l=sys.argv[1]
for r in csv.DictReader(open(f'logs/h71b-{l}.csv')):
    print(f"{l}: n_prompt={r['n_prompt']} n_gen={r['n_gen']} avg_ts={float(r['avg_ts']):.2f} sd={float(r['stddev_ts']):.2f}")
PY
    return $rc; }

bench off GGML_PASCAL_MMVQ_STAGE=0; bench on
python3 - <<'PY' > results/h71b-bench.txt
import csv
g = {l: {('pp4' if r['n_gen']=='0' else 'tg32'): float(r['avg_ts']) for r in csv.DictReader(open(f'logs/h71b-{l}.csv'))} for l in ('off','on')}
for t in ('tg32','pp4'): print(f"{t}: off {g['off'][t]:.2f} on {g['on'][t]:.2f} ({100*(g['on'][t]/g['off'][t]-1):+.1f}%)")
print('GO' if g['on']['tg32'] >= 1.015*g['off']['tg32'] else 'STOP')
PY
cat results/h71b-bench.txt; fin bench
grep -q '^GO' results/h71b-bench.txt || { echo "STOP: tg32 gain < 1.5%"; exit 0; }

export N_PREDICT=400 BATCH=2048 UB=2048 BIN=$FORK/llama-server REPS=3 SPLIT=tensor
cool; CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt ./scripts/run-spec-placement.sh h71b-q6stage-16k draft-mtp "$MTP" default
steps h58-f16v-16k; steps h71b-q6stage-16k; cmp=$(python3 scripts/compare-outputs.py h58-f16v-16k h71b-q6stage-16k); echo "$cmp"; fin 16k
if ! echo "$cmp" | grep -q identical; then
  cool; KV=f16 SPEC=mtp ./scripts/run-phase8-tier2.sh h71b-q6stage-f16
  hs=$(awk -F, '$1=="h71b-q6stage-f16"{c+=$5} END{print c+0}' results/phase8-tier2.csv); echo "haystack: $hs/19 (h58 19/19)"; fin haystack
else echo "output identical: haystack skipped"; fi
echo "=== H71b done ==="
