#!/usr/bin/env bash
# H27 (fork D1): CUDA graphs on sm_60, off vs on, 2-card and 1-card, drafter-free at 16k.
# Plus RUNBOOK §7 item 6 (D5): GGML_CUDA_P2P=1 with MTP vs banked h24-ub2048-mtp-16k.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FORK=/root/llama.cpp-Pascal/build-cuda-p100/bin/llama-server
export GGML_CUDA_ALLREDUCE=none REPS=2 N_PREDICT=400 CTX=16384 \
       PROMPT_FILE=prompts/h11-depth-prompt.txt BATCH=2048 UB=2048

cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 60 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20
    done
}
arm() {  # label graphs split model
    cool
    GGML_CUDA_PASCAL_GRAPHS=$2 SPLIT=$3 MODEL=$4 BIN=$FORK \
        ./scripts/run-spec-placement.sh "$1" none none default
    echo "--- $1 done (exit $?) ---"; sleep 15
}

Q4=/root/Qwen3.8-27B-UD-Q4_K_M.gguf; IQ3=/root/Qwen3.8-27B-UD-IQ3_S.gguf
arm h27-2card-graphs0 0 tensor "$Q4"
arm h27-2card-graphs1 1 tensor "$Q4"
arm h27-1card-graphs0 0 none "$IQ3"
arm h27-1card-graphs1 1 none "$IQ3"

cool
GGML_CUDA_P2P=1 SPLIT=tensor ./scripts/run-spec-placement.sh d5-p2p-mtp-16k draft-mtp \
    /root/mtp-Qwen3.8-27B-Q4_0.gguf default
echo "--- d5-p2p-mtp-16k done (exit $?) ---"

echo "=== H27 / D5 results ==="
grep -E '^(label|h27-|d5-|h24-ub2048)' results/h11-placement.csv | column -s, -t
for c in 2card 1card; do
    a=results/h11/h27-$c-graphs0.jsonl b=results/h11/h27-$c-graphs1.jsonl
    [[ -s $a && -s $b ]] && python3 - "$a" "$b" "$c" <<'PY'
import json, sys
txt = lambda f: [json.loads(l)["choices"][0]["message"]["content"] for l in open(f)]
a, b = txt(sys.argv[1]), txt(sys.argv[2])
print(f"{sys.argv[3]}: greedy output graphs0 vs graphs1 identical: {a == b} (reps {len(a)}/{len(b)})")
PY
done

git add -A results logs HYPOTHESES.md scripts/run-h27-graphs.sh
git commit -q -m "bench: H27 CUDA graphs on sm_60 (2-card + 1-card) and D5 P2P with MTP at 16k" \
    && { git push -q && echo pushed || echo "WARNING: push failed; commit is local" >&2; }
