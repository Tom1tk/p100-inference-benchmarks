#!/usr/bin/env bash
# Build the rebased DFlash2 tree, but only once the tensor cells are done --
# a -j12 compile alongside a -t 8 benchmark would skew the benchmark.
set -uo pipefail
while pgrep -f 'run-tensor-cells.sh|llama-bench' >/dev/null; do sleep 30; done
echo "=== benches clear, building $(date -Is) ==="
cd /root/dflash2-rebased
cmake -B build-cuda-p100 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 \
    -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF 2>&1 | tail -5
cmake --build build-cuda-p100 --config Release -j12 2>&1 | tail -25
echo "=== build exit=$? ==="
ls -la build-cuda-p100/bin/llama-bench 2>/dev/null || echo "NO BINARY"
