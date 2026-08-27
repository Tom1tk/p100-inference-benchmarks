#!/usr/bin/env bash
# H25: how does Qwen3.8-27B-UD-IQ3_S on ONE P100 compare to the two-card pair?
#
# Short context only (<=16k) so every arm is minutes, not the 30 min a 100k
# prefill costs. Levers swept: split mode, ubatch, KV cache type, MTP on/off.
#
# NOTE on -sm: with one visible device the split mode is a no-op. The two-card
# tensor/layer arms are here as the comparison baseline for the same quant, so
# "IQ3 vs Q4_K_M" is separated from "one card vs two".
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IQ3=/root/Qwen3.8-27B-UD-IQ3_S.gguf
SWEEP=./scripts/run-ubatch-sweep.sh

cool() {
    for _ in $(seq 1 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 70 ]] && return
        echo "cooling: hottest card ${t}C"; sleep 20
    done
}

# --- llama-bench arms: prefill + decode, no drafter -------------------------
# label                   devices        split   ubatches   ctk/ctv  prompts
ARMS=(
  "h25-iq3-1card          CUDA0          none    512,2048   f16      4096,16384"
  "h25-iq3-2card-tensor   CUDA0,CUDA1    tensor  512,2048   f16      4096,16384"
  "h25-iq3-2card-layer    CUDA0,CUDA1    layer   512,2048   f16      4096,16384"
  "h25-iq3-1card-kv-q8    CUDA0          none    2048       q8_0     16384"
  "h25-iq3-1card-kv-q4    CUDA0          none    2048       q4_0     16384"
)

for arm in "${ARMS[@]}"; do
    read -r label devices split ubs kv prompts <<<"$arm"
    cool
    MODEL="$IQ3" LABEL="$label" DEVICES="$devices" SPLIT="$split" \
    UBATCHES="$ubs" CTK="$kv" CTV="$kv" PROMPTS="$prompts" \
    BATCH=2048 GEN=128 REPS=2 "$SWEEP"
    echo "--- $label done (exit $?) ---"
    sleep 10
done

# --- server arms: MTP on/off on one card at 16k -----------------------------
# The drafter is another 1306 MiB on a card that already holds 11483 MiB of
# weights. If this OOMs, that is the answer to "can the fallback use MTP".
for spec in mtp none; do
    cool
    if [[ "$spec" == mtp ]]; then
        args=(draft-mtp /root/mtp-Qwen3.8-27B-Q4_0.gguf default)
    else
        args=(none none default)
    fi
    MODEL="$IQ3" SPLIT=none CTX=16384 REPS=3 N_PREDICT=400 \
    PROMPT_FILE=prompts/h11-depth-prompt.txt BATCH=2048 UB=2048 \
    ./scripts/run-spec-placement.sh "h25-iq3-1card-${spec}-16k" "${args[@]}"
    echo "--- h25 ${spec} done (exit $?) ---"
    sleep 10
done

echo "=== H25 server arms ==="
grep -E '^(label|h25-)' results/h11-placement.csv | column -s, -t

git add -A
if git diff --cached --quiet; then
    echo "nothing to commit"
else
    git commit -q -m "bench: h25 IQ3_S single-card vs pair at <=16k (split, ubatch, KV type, MTP)"
    git push -q && echo "pushed" || echo "WARNING: push failed; commit is local" >&2
fi
