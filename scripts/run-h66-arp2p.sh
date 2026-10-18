#!/usr/bin/env bash
# H66: one-kernel P2P AllReduce for decode-sized exchanges (GGML_PASCAL_AR_P2P, default on sm_60).
# 1) llama-bench tg32 + pp4, lever 0 vs default, -r 5. Stop if tg32 gains < 1.5%.
# 2) 16k serve command REPS=3 vs h56-rebase-16k. 3) haystack (f16 KV, MTP), gate >= 18/19 (h57 19/19).
# 4) 98k REPS=3 vs r10s-fork-100k (ms per target step).
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
fin() { git add -A results logs scripts/run-h66-arp2p.sh
    git commit -q -m "bench: H66 P2P AllReduce ($1)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }; }
steps() { awk -F, -v l="$1" '$1==l && $16=="ok"{d+=$6; n++; s+=($8-$10); t+=$8/$6; a+=$11} END{if(n) printf "%s: decode %.2f, accept %.1f%%, %.2f ms/target-step (n=%d)\n", l, d/n, a/n, 1000*t/s, n}' results/h11-placement.csv; }
bench() { cool; ABORT_TEMP=80 INTERVAL=2 ./scripts/gpu-monitor.sh > "logs/h66-$1-temps.log" 2>/dev/null & local mon=$!
    env CUDA_VISIBLE_DEVICES=0,1 "${@:2}" $FORK/llama-bench -m /root/Qwen3.8-27B-UD-Q4_K_M.gguf -sm tensor -ngl 99 -fa 1 -t 8 \
        -p 4 -n 32 -r 5 -o csv > "logs/h66-$1.csv" 2> "logs/h66-$1.log"; local rc=$?; kill $mon
    grep -c "one-kernel P2P AllReduce" "logs/h66-$1.log" | sed "s/^/$1 engaged: /"
    python3 - "$1" <<'PY'
import csv,sys; l=sys.argv[1]
for r in csv.DictReader(open(f'logs/h66-{l}.csv')):
    print(f"{l}: n_prompt={r['n_prompt']} n_gen={r['n_gen']} avg_ts={float(r['avg_ts']):.2f} sd={float(r['stddev_ts']):.2f}")
PY
    return $rc; }

bench off GGML_PASCAL_AR_P2P=0; bench on
python3 - <<'PY' > results/h66-bench.txt
import csv
g = {l: {('pp4' if r['n_gen']=='0' else 'tg32'): float(r['avg_ts']) for r in csv.DictReader(open(f'logs/h66-{l}.csv'))} for l in ('off','on')}
for t in ('tg32','pp4'): print(f"{t}: off {g['off'][t]:.2f} on {g['on'][t]:.2f} ({100*(g['on'][t]/g['off'][t]-1):+.1f}%)")
print('GO' if g['on']['tg32'] >= 1.015*g['off']['tg32'] else 'STOP')
PY
cat results/h66-bench.txt; fin bench
grep -q '^GO' results/h66-bench.txt || { echo "STOP: tg32 gain < 1.5%"; exit 0; }

export N_PREDICT=400 BATCH=2048 UB=2048 BIN=$FORK/llama-server REPS=3 SPLIT=tensor
cool; CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt ./scripts/run-spec-placement.sh h66-arp2p-16k draft-mtp "$MTP" default
steps h56-rebase-16k; steps h66-arp2p-16k; python3 scripts/compare-outputs.py h56-rebase-16k h66-arp2p-16k; fin 16k
cool; KV=f16 SPEC=mtp ./scripts/run-phase8-tier2.sh h66-arp2p-f16
hs=$(awk -F, '$1=="h66-arp2p-f16"{c+=$5} END{print c+0}' results/phase8-tier2.csv); echo "haystack: $hs/19 (h57 19/19)"; fin haystack
[[ "$hs" -ge 18 ]] || { echo "STOP: haystack below gate"; exit 0; }
cool; CTX=100000 NP=1 REQ_TIMEOUT=1800 PROMPT_FILE=prompts/h26-100k-prompt.txt ./scripts/run-spec-placement.sh h66-arp2p-98k draft-mtp "$MTP" default
steps r10s-fork-100k; steps h66-arp2p-98k; fin 98k
echo "=== H66 done ==="
