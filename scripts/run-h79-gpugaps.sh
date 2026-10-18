#!/usr/bin/env bash
# H79-H81: GPU kernel gaps, stacked one lever per llama-bench arm (tg32 + pp4 -r 5):
#   off   all three off                      control
#   cpy   + H79 cpy fastdiv                  (vs off)
#   iq4   + H80 IQ4_NL reference mmvq        (vs cpy)
#   all   + H81 GDN one column per warp      (vs iq4) = build default
# Then 16k serve REPS=3 vs banked h76-on-16k if all beats off by >= 0.8% on tg32; haystack if output differs.
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
fin() { git add -A results logs scripts/run-h79-gpugaps.sh
    git commit -q -m "bench: H79-H81 GPU kernel gaps ($1)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }; }
steps() { awk -F, -v l="$1" '$1==l && $16=="ok"{d+=$6; n++; s+=($8-$10); t+=$8/$6; a+=$11} END{if(n) printf "%s: decode %.2f, accept %.1f%%, %.2f ms/target-step (n=%d)\n", l, d/n, a/n, 1000*t/s, n}' results/h11-placement.csv; }
bench() { local l=$1; shift
    cool; ABORT_TEMP=80 INTERVAL=2 ./scripts/gpu-monitor.sh > "logs/h79-$l-temps.log" 2>/dev/null & local mon=$!
    env CUDA_VISIBLE_DEVICES=0,1 "$@" $FORK/llama-bench -m /root/Qwen3.8-27B-UD-Q4_K_M.gguf -sm tensor -ngl 99 -fa 1 -t 8 \
        -p 4 -n 32 -r 5 -o csv > "logs/h79-$l.csv" 2> "logs/h79-$l.log"; local rc=$?; kill $mon
    [[ $rc -eq 0 ]] || { echo "ERROR: $l llama-bench rc=$rc"; fin "$l failed"; exit 1; }; }

bench off GGML_PASCAL_CPY_FASTDIV=0 GGML_PASCAL_MMVQ_REF_IQ4NL=0 GGML_PASCAL_GDN_COL1=0
bench cpy GGML_PASCAL_MMVQ_REF_IQ4NL=0 GGML_PASCAL_GDN_COL1=0
bench iq4 GGML_PASCAL_GDN_COL1=0
bench all
python3 - <<'PY' > results/h79-bench.txt
import csv
A = ('off', 'cpy', 'iq4', 'all')
g = {l: {('pp4' if r['n_gen']=='0' else 'tg32'): (float(r['avg_ts']), float(r['stddev_ts'])) for r in csv.DictReader(open(f'logs/h79-{l}.csv'))} for l in A}
for t in ('tg32','pp4'):
    print(t, '  '.join(f"{l} {g[l][t][0]:.2f}±{g[l][t][1]:.2f}" for l in A))
    print('  ' + '  '.join(f"{h} {100*(g[b][t][0]/g[a][t][0]-1):+.2f}%" for h, a, b in (('H79','off','cpy'), ('H80','cpy','iq4'), ('H81','iq4','all'), ('stack','off','all'))))
print('GO' if g['all']['tg32'][0] >= 1.008*g['off']['tg32'][0] else 'STOP')
PY
cat results/h79-bench.txt; fin bench
grep -q '^GO' results/h79-bench.txt || { echo "STOP: stack < +0.8% tg32"; exit 0; }

export N_PREDICT=400 BATCH=2048 UB=2048 BIN=$FORK/llama-server REPS=3 SPLIT=tensor
cool; CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt ./scripts/run-spec-placement.sh h79-gpugaps-16k draft-mtp "$MTP" default
steps h76-on-16k; steps h79-gpugaps-16k; cmp=$(python3 scripts/compare-outputs.py h76-on-16k h79-gpugaps-16k); echo "$cmp"; fin 16k
if ! echo "$cmp" | grep -q identical; then
  cool; KV=f16 SPEC=mtp ./scripts/run-phase8-tier2.sh h79-gpugaps-f16
  hs=$(awk -F, '$1=="h79-gpugaps-f16"{c+=$5} END{print c+0}' results/phase8-tier2.csv); echo "haystack: $hs/19 (h58 19/19)"; fin haystack
else echo "output identical: haystack skipped"; fi
echo "=== H79 done ==="
