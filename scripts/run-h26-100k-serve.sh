#!/usr/bin/env bash
# H26: does the deliverable config actually serve at 100k?
#
# Everything in this repo's serve-side record is 16k. H13 measured pp100000 on
# llama-bench only -- no drafter, no server, no decode, no VRAM, no acceptance,
# no sustained thermal profile. This run answers all of those at once, because
# they all come free from the same request.
#
# Arms are a LADDER, not a sweep: each one only runs if the one above it failed.
# The first arm that works is the answer, and the rest are not worth the power.
#
#   A  q8_0 KV + MTP    the config we want. If it works, stop.
#   B  q4_0 KV + MTP    only if A cannot fit. H25 says q4_0 costs ~nothing over
#                       q8_0 on speed, so this is purely a VRAM fallback.
#   C  q8_0 KV, no MTP  only if both fit-attempts fail. Establishes whether 100k
#                       is reachable at all without a drafter.
#
# 100k prefill is ~8 min of SUSTAINED compute against an 83C abort limit, and
# nothing in this repo has ever run hot for more than ~3 min. Watch
# logs/*.temps.log while this runs; do not walk away from it.

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# -c 100000 matches H13's measurement point. The prompt is 97,620 tokens, so
# with 400 generated there are ~1,900 tokens of slack.
export SPLIT=tensor GGML_CUDA_ALLREDUCE=none \
       REPS=2 N_PREDICT=400 CTX=100000 \
       PROMPT_FILE=prompts/h26-100k-prompt.txt \
       BATCH=2048 UB=2048 \
       NP=1 REQ_TIMEOUT=1800

MTP=/root/mtp-Qwen3.8-27B-Q4_0.gguf

run_arm() {  # label ctk ctv spec drafter
    export CTK="$2" CTV="$3"
    echo
    echo "########## $1 (KV $2/$3, drafter ${4}) ##########"
    ./scripts/run-spec-placement.sh "$1" "$4" "$5" default
}

# Arm A -- the one we actually want.
if run_arm h26-100k-mtp-kvq8 q8_0 q8_0 draft-mtp "$MTP"; then
    echo "ARM A SUCCEEDED: q8_0 KV + MTP serves at 100k. Skipping arms B and C."
    exit 0
fi

echo "Arm A failed. Falling back to q4_0 KV to buy VRAM." >&2
if run_arm h26-100k-mtp-kvq4 q4_0 q4_0 draft-mtp "$MTP"; then
    echo "ARM B SUCCEEDED: 100k needs q4_0 KV to fit a drafter. Skipping arm C."
    exit 0
fi

echo "Both drafter arms failed. Arm C asks whether 100k works with no drafter." >&2
run_arm h26-100k-none-kvq8 q8_0 q8_0 none none
