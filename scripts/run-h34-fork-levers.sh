#!/usr/bin/env bash
# H31 + H33 + H34 on one rebuilt fork binary (all three levers env-gated, default off):
#   H34 GGML_CUDA_PASCAL_MMVH=1 now also covers Q6_K   -> decode only (n <= 8)
#   H33 GGML_CUDA_PASCAL_GEMM_ALGO=1 (cuBLAS ALGO3)   -> prefill only (n >= 512)
#   H31 GGML_PASCAL_F16_REDUCE=1                      -> 2-card prefill only (reduces >= 1 MiB)
# H33 and H34 touch disjoint paths (n >= 512 vs n <= 8), so one arm measures both: its decode
# against the banked control is H34, its prefill is H33. H31 then adds F16_REDUCE on top of that
# arm (one lever). Correctness and KLD gates first; a lever that fails its gate is left out.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FORKBIN=/root/llama.cpp-Pascal/build-cuda-p100/bin
Q4=/root/Qwen3.8-27B-UD-Q4_K_M.gguf
IQ3=/root/Qwen3.8-27B-UD-IQ3_S.gguf
KLDIR=/root/h28-kld; mkdir -p "$KLDIR" logs results/h34
CSV=results/h34-kld.csv
export GGML_CUDA_ALLREDUCE=none

cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 60 ]] && return; echo "cooling: hottest card ${t}C"; sleep 20
    done
}
ppl() {  # label split ub env-assignments... -- extra-args...
    local label=$1 split=$2 ub=$3; shift 3
    local envs=(); while [[ "$1" != -- ]]; do envs+=("$1"); shift; done; shift
    cool
    ABORT_TEMP=80 INTERVAL=2 ./scripts/gpu-monitor.sh > "logs/h34-$label-temps.log" 2>/dev/null &
    local mon=$!
    env "${envs[@]}" "$FORKBIN/llama-perplexity" -m "$Q4" -f prompts/h26-100k-prompt.txt \
        -c 512 --chunks 8 -b 512 -ub "$ub" -ngl 99 -fa 1 -sm "$split" "$@" > "logs/h34-$label.log" 2>&1
    local rc=$?
    kill $mon
    echo "--- $label rc=$rc, peak $(tr '|' '\n' < "logs/h34-$label-temps.log" | awk -F', ' 'NF>=3{print $2}' | sort -n | tail -1)C ---"
    return $rc
}
kld() {  # same as ppl, then appends the KLD row
    ppl "$@" && { local f="logs/h34-$1.log"
        printf '%s,%s,%s,%s\n' "$1" "$(grep -m1 'Mean    KLD:' "$f" | awk '{print $3}')" \
            "$(grep -m1 '99.0%   KLD:' "$f" | awk '{print $3}')" "$(grep -m1 'Same top p:' "$f" | awk '{print $4}')" >> "$CSV"; }
}

# 0. Correctness: Q6_K through mmvh vs the CPU backend
GGML_CUDA_PASCAL_MMVH=1 "$FORKBIN/test-backend-ops" test -o MUL_MAT -b CUDA0 -p 'type_a=q6_K' \
    > logs/h34-test-backend-ops.log 2>&1
grep -E 'tests passed|FAIL' logs/h34-test-backend-ops.log | tail -3
grep -q FAIL logs/h34-test-backend-ops.log && { echo "Q6_K mmvh fails test-backend-ops"; MMVH_OK=0; } || MMVH_OK=1

# 1. KLD gates. Bases are deterministic, so each lever is compared to its own all-off base.
echo "label,mean_kld,p99_kld,same_top_pct" > "$CSV"
# H34: same config as banked H28 (-sm layer, ub 8): mmvh-ub8 there was mean 0.021738, same-top 99.167
if [[ $MMVH_OK == 1 ]]; then
    ppl base-ub8 layer 8 GGML_CUDA_PASCAL_MMVH=0 -- --kl-divergence-base "$KLDIR/h34-ub8.kld" \
        && kld q6k-mmvh-ub8 layer 8 GGML_CUDA_PASCAL_MMVH=1 -- --kl-divergence-base "$KLDIR/h34-ub8.kld" --kl-divergence
fi
# H33 and H31: -sm tensor, ub 512 (every GEMM takes the n >= 512 path, every reduce goes F16)
ppl base-ub512 tensor 512 GGML_CUDA_P2P=1 -- --kl-divergence-base "$KLDIR/h34-ub512.kld" || exit 1
kld gemm-algo-ub512 tensor 512 GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_GEMM_ALGO=1 -- \
    --kl-divergence-base "$KLDIR/h34-ub512.kld" --kl-divergence
kld f16-reduce-ub512 tensor 512 GGML_CUDA_P2P=1 GGML_PASCAL_F16_REDUCE=1 -- \
    --kl-divergence-base "$KLDIR/h34-ub512.kld" --kl-divergence
rm -f "$KLDIR"/h34-*.kld
column -s, -t "$CSV"

read -r Q6K ALGO F16 < <(python3 - "$CSV" <<'PY'
import csv, sys
r = {x["label"]: x for x in csv.DictReader(open(sys.argv[1]))}
def ok(label, mean_max, top_min):
    m = r.get(label)
    return "1" if m and m["mean_kld"] and float(m["mean_kld"]) <= mean_max and float(m["same_top_pct"]) >= top_min else "0"
print(ok("q6k-mmvh-ub8", 0.026, 98.9), ok("gemm-algo-ub512", 0.002, 99.0), ok("f16-reduce-ub512", 0.002, 99.0))
PY
)
echo "gates: q6k=$Q6K gemm-algo=$ALGO f16-reduce=$F16"

# every speed arm runs MMVH=1, so a failed Q6_K gate needs a fix before any of them
[[ $Q6K == 1 ]] || { echo "Q6_K gate failed: no speed arms"; ALGO=0; F16=0; }

# 2. Speed arms, 16k depth, drafter-free. Controls (banked): h28-2card-mmvh1 25.29 / 342.25,
#    h29-1card-mmvh-iq3 15.86 / 196.40 (rep 2, decode t/s / prefill t/s).
export REPS=2 N_PREDICT=400 CTX=16384 PROMPT_FILE=prompts/h11-depth-prompt.txt BATCH=2048 UB=2048 \
       BIN=$FORKBIN/llama-server GGML_CUDA_PASCAL_MMVH=1 GGML_CUDA_PASCAL_GEMM_ALGO=$ALGO
arm() { cool; "$@"; echo "--- done (exit $?) ---"; sleep 15; }
if [[ $Q6K == 1 ]]; then
    arm env GGML_CUDA_P2P=0 SPLIT=tensor MODEL="$Q4" ./scripts/run-spec-placement.sh h34-2card none none default
    arm env SPLIT=none MODEL="$IQ3" ./scripts/run-spec-placement.sh h34-1card-iq3 none none default
fi
if [[ $F16 == 1 ]]; then
    # 1-card arm skipped (rule zero): the reduce never runs on one card
    arm env GGML_CUDA_P2P=0 GGML_PASCAL_F16_REDUCE=1 SPLIT=tensor MODEL="$Q4" \
        ./scripts/run-spec-placement.sh h34-2card-f16reduce none none default
fi
echo "=== results ==="
grep -E '^(label|h28-2card-mmvh1,|h29-1card-mmvh-iq3|h34-)' results/h11-placement.csv | column -s, -t

git add -A results logs scripts/run-h34-fork-levers.sh
git commit -q -m "bench: H31/H33/H34 fork levers (F16 reduce, GEMM ALGO3, Q6_K mmvh) gates and 16k arms" \
    && { git push -q && echo pushed || echo "WARNING: push failed; commit is local" >&2; }
