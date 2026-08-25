#!/usr/bin/env bash
# H10 remaining half: does GGML_CUDA_P2P=1 help -sm tensor?
#
# Peer access is off unless GGML_CUDA_P2P is set. These cards peer in both
# directions (cudaDeviceCanAccessPeer = 1, PHB, single NUMA node), and every
# tensor-split run on P100 goes through the meta-backend butterfly AllReduce
# (the internal pipeline is Volta-gated -- see H10 Finding 3), which stages
# cross-GPU traffic through host memory when peering is off.
#
# Tensor split moves far more cross-GPU data than layer split, so this is the
# best available test bed. Everything but the env var is fixed.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Don't compete for CPU with a running matrix.
while pgrep -f 'run-phase2-tensor.sh|llama-server|llama-bench' >/dev/null; do sleep 30; done

cool() {
    for _ in $(seq 30); do
        t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        [[ "$t" -le 70 ]] && return 0
        echo "cooling: hottest ${t}C"; sleep 20
    done
}

for p2p in off on; do
    cool
    echo "########## p2p=$p2p"
    if [[ "$p2p" == on ]]; then export GGML_CUDA_P2P=1; else unset GGML_CUDA_P2P; fi
    GGML_CUDA_ALLREDUCE=none ./scripts/run-bench.sh mainline-rebased \
        /root/Qwen3.8-27B-UD-Q4_K_M.gguf tensor "h10-p2p-${p2p}-q4km" 2>&1 | tail -12
done

echo "=== h10 p2p complete ==="
