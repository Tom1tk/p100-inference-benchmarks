#!/usr/bin/env bash
# H28 follow-up: which mmvh type causes the KLD tail? Needs the fork's DIAG switch
# (GGML_CUDA_PASCAL_MMVH=2 excludes IQ4_XS, =3 is IQ4_XS only). Same KLD setup as run-h28-mmvh.sh.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PPL=/root/llama.cpp-Pascal/build-cuda-p100/bin/llama-perplexity
Q4=/root/Qwen3.8-27B-UD-Q4_K_M.gguf; KLDIR=/root/h28-kld; CSV=results/h28-kld-split.csv
mkdir -p "$KLDIR"
ABORT_TEMP=80 INTERVAL=2 ./scripts/gpu-monitor.sh > logs/h28-split-temps.log 2>/dev/null &
MON=$!
run() {  # label mmvh extra...
    local l=$1 m=$2; shift 2
    GGML_CUDA_PASCAL_MMVH=$m "$PPL" -m "$Q4" -f prompts/h26-100k-prompt.txt -c 512 --chunks 8 -b 512 -ub 8 \
        -ngl 99 -fa 1 -sm layer --kl-divergence-base "$KLDIR/base.kld" "$@" > "logs/h28-split-$l.log" 2>&1
    echo "--- $l rc=$? ---"
}
echo "label,mean_kld,p99_kld,p999_kld,max_kld,same_top_pct" > "$CSV"
run base 0
for a in "no-iq4xs 2" "iq4xs-only 3"; do
    set -- $a; run "$1" "$2" --kl-divergence; f=logs/h28-split-$1.log
    printf '%s,%s,%s,%s,%s,%s\n' "$1" "$(grep -m1 'Mean    KLD:' $f | awk '{print $3}')" "$(grep -m1 '99.0%   KLD:' $f | awk '{print $3}')" \
        "$(grep -m1 '99.9%   KLD:' $f | awk '{print $3}')" "$(grep -m1 'Maximum KLD:' $f | awk '{print $3}')" "$(grep -m1 'Same top p:' $f | awk '{print $4}')" >> "$CSV"
done
kill $MON; rm -f "$KLDIR/base.kld"
column -s, -t "$CSV"; echo "peak $(tr '|' '\n' < logs/h28-split-temps.log | awk -F', ' 'NF>=3{print $2}' | sort -n | tail -1)C"
git add -A results logs scripts/run-h28-kld-split.sh
git commit -q -m "bench: H28 KLD tail split by weight type" && { git push -q && echo pushed; }
