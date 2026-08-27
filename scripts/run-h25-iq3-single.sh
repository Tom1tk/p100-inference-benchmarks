#!/usr/bin/env bash
# H25: Qwen3.8-27B-UD-IQ3_S on ONE P100 - how does it compare to the pair?
#
# Single GPU only (-dev CUDA0), so -sm is a no-op and is not swept; the two-card
# numbers already exist (H13/H14/H24 on Q4_K_M). Short context only (<=16k).
#
# KV is q8_0 or q4_0, never f16: f16 is the native path on sm_60 and the
# single-card f16 baseline is already banked in results/raw/h25-iq3-1card.csv,
# so these arms measure what quantising KV costs and what depth it buys.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IQ3=/root/Qwen3.8-27B-UD-IQ3_S.gguf

cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 70 ]] && return
        echo "cooling: hottest card ${t}C"; sleep 20
    done
}

# --- Arms 1 and 2: KV type x ubatch, prefill and decode, no drafter ---------
# q8_0 runs first: if quantised KV is unsupported on sm_60 it fails at pp4096
# in ~3 min rather than after a full arm.
#
# -ub 512 is deliberately NOT swept here. H14 and the banked f16 arm both show
# 2048 wins on prefill, and H24 plus that same arm show ubatch does not touch
# decode at all. Re-running it would buy a number we already have. Each run
# costs real electricity - do not re-measure a settled lever.
# label                 ctk/ctv  ubatches
ARMS=(
  "h25-iq3-1card-kv-q8  q8_0     2048"
  "h25-iq3-1card-kv-q4  q4_0     2048"
)

for arm in "${ARMS[@]}"; do
    read -r label kv ubs <<<"$arm"
    cool
    MODEL="$IQ3" LABEL="$label" DEVICES=CUDA0 SPLIT=none \
    UBATCHES="$ubs" CTK="$kv" CTV="$kv" PROMPTS=4096,16384 \
    BATCH=2048 GEN=128 REPS=2 ./scripts/run-ubatch-sweep.sh
    status=$?
    echo "--- $label done (exit $status) ---"
    # Early abort: q4_0 uses the same quantised-KV attention path as q8_0. If
    # q8_0 could not run, q4_0 cannot either, and burning a second load to
    # confirm it teaches us nothing.
    if [[ "$status" -ne 0 && "$kv" == q8_0 ]]; then
        echo "SKIPPING the q4_0 arm: quantised KV failed on q8_0, same code path" >&2
        break
    fi
    sleep 10
done

# --- Arm 3: MTP on vs off at 16k, one card, q8_0 ----------------------------
# The drafter is another 1306 MiB on a card already holding 11483 MiB of
# weights. An OOM here is the answer to "can the fallback use MTP at all".
# The 'none' leg runs even if 'mtp' OOMs: it is the matched single-card decode
# control at 16k, which nothing else in the repo has, so it is never redundant.
for spec in mtp none; do
    cool
    if [[ "$spec" == mtp ]]; then
        args=(draft-mtp /root/mtp-Qwen3.8-27B-Q4_0.gguf default)
    else
        args=(none none default)
    fi
    MODEL="$IQ3" SPLIT=none CTK=q8_0 CTV=q8_0 CTX=16384 REPS=2 N_PREDICT=400 \
    PROMPT_FILE=prompts/h11-depth-prompt.txt BATCH=2048 UB=2048 \
    ./scripts/run-spec-placement.sh "h25-iq3-1card-${spec}-16k" "${args[@]}"
    echo "--- arm3 ${spec} done (exit $?) ---"
    sleep 10
done

echo "=== H25 arm 3 ==="
grep -E '^(label|h25-)' results/h11-placement.csv | column -s, -t

git add -A
if git diff --cached --quiet; then
    echo "nothing to commit"
else
    git commit -q -m "bench: h25 IQ3_S single-card at <=16k (KV q8_0/q4_0, ubatch, MTP)"
    git push -q && echo "pushed" || echo "WARNING: push failed; commit is local" >&2
fi
