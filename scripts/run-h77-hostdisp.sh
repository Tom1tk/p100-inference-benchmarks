#!/usr/bin/env bash
# H77/H78: host dispatch overhead. llama-bench tg32 + pp4 -r 5, three arms:
#   h76     H76 build (/root/p100-builds/h76-bin), default            control
#   h76pdl0 H76 build, GGML_CUDA_PDL=0                                 H78 (vs h76)
#   new     new build: can_fuse early-out + PDL lookup off pre-Hopper  H77 (vs h76pdl0)
# Then 16k serve REPS=3 vs banked h76-on-16k if new beats h76 by >= 1% on tg32; output must be byte-identical.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unset $(env | grep -oE '^GGML_(CUDA_)?PASCAL_[A-Z0-9_]+' || true)
FORK=/root/llama.cpp-Pascal/build-cuda-p100/bin OLD=/root/p100-builds/h76-bin MTP=/root/mtp-Qwen3.8-27B-Q4_0-head96k.gguf
export GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader | paste -sd' ')
echo "cap: $cap"; [[ "$cap" == "200.00 W 200.00 W" ]] || { echo "ERROR: expected 200 W on both cards"; exit 1; }
cool() { for _ in $(seq 1 30); do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
    [[ "$t" -le 55 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20; done; }
fin() { git add -A results logs scripts/run-h77-hostdisp.sh
    git commit -q -m "bench: H77/H78 host dispatch ($1)" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }; }
steps() { awk -F, -v l="$1" '$1==l && $16=="ok"{d+=$6; n++; s+=($8-$10); t+=$8/$6; a+=$11} END{if(n) printf "%s: decode %.2f, accept %.1f%%, %.2f ms/target-step (n=%d)\n", l, d/n, a/n, 1000*t/s, n}' results/h11-placement.csv; }
bench() { local l=$1 bin=$2; shift 2
    cool; ABORT_TEMP=80 INTERVAL=2 ./scripts/gpu-monitor.sh > "logs/h77-$l-temps.log" 2>/dev/null & local mon=$!
    env CUDA_VISIBLE_DEVICES=0,1 "$@" $bin/llama-bench -m /root/Qwen3.8-27B-UD-Q4_K_M.gguf -sm tensor -ngl 99 -fa 1 -t 8 \
        -p 4 -n 32 -r 5 -o csv > "logs/h77-$l.csv" 2> "logs/h77-$l.log"; local rc=$?; kill $mon
    [[ $rc -eq 0 ]] || { echo "ERROR: $l llama-bench rc=$rc"; fin "$l failed"; exit 1; }; }

bench h76 $OLD LD_LIBRARY_PATH=$OLD
bench h76pdl0 $OLD LD_LIBRARY_PATH=$OLD GGML_CUDA_PDL=0
bench new $FORK
python3 - <<'PY' > results/h77-bench.txt
import csv
g = {l: {('pp4' if r['n_gen']=='0' else 'tg32'): (float(r['avg_ts']), float(r['stddev_ts'])) for r in csv.DictReader(open(f'logs/h77-{l}.csv'))}
     for l in ('h76','h76pdl0','new')}
for t in ('tg32','pp4'):
    print(t, '  '.join(f"{l} {v[t][0]:.2f}±{v[t][1]:.2f}" for l, v in g.items()))
    print(f"  H78 pdl0/h76 {100*(g['h76pdl0'][t][0]/g['h76'][t][0]-1):+.2f}%  H77 new/pdl0 {100*(g['new'][t][0]/g['h76pdl0'][t][0]-1):+.2f}%"
          f"  total new/h76 {100*(g['new'][t][0]/g['h76'][t][0]-1):+.2f}%")
print('GO' if g['new']['tg32'][0] >= 1.01*g['h76']['tg32'][0] else 'STOP')
PY
cat results/h77-bench.txt; fin bench
grep -q '^GO' results/h77-bench.txt || { echo "STOP: new build < +1% tg32 over H76"; exit 0; }

export N_PREDICT=400 BATCH=2048 UB=2048 BIN=$FORK/llama-server REPS=3 SPLIT=tensor
cool; CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt ./scripts/run-spec-placement.sh h77-hostdisp-16k draft-mtp "$MTP" default
steps h76-on-16k; steps h77-hostdisp-16k; cmp=$(python3 scripts/compare-outputs.py h76-on-16k h77-hostdisp-16k); echo "$cmp"; fin 16k
if ! echo "$cmp" | grep -q identical; then
  cool; KV=f16 SPEC=mtp ./scripts/run-phase8-tier2.sh h77-hostdisp-f16
  hs=$(awk -F, '$1=="h77-hostdisp-f16"{c+=$5} END{print c+0}' results/phase8-tier2.csv); echo "haystack: $hs/19 (h58 19/19)"; fin haystack
else echo "output identical: haystack skipped"; fi
echo "=== H77 done ==="
