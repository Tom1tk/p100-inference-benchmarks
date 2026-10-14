#!/bin/bash
# Phase 0.2: nsys trace of one 2048-token prefill + 64 decode tokens (llama-bench, r=1,
# with warmup), 2-card Q4_K_M -sm tensor and 1-card IQ3_S -sm none. Live 80 C abort.
B=/root/llama.cpp-Pascal/build-cuda-p100/bin/llama-bench
OUT=/root/p100-benchmarks/results/phase0/nsys
mkdir -p $OUT
COMMON="-ngl 99 -fa 1 -t 8 -ctk f16 -ctv f16 -b 2048 -ub 2048 -p 2048 -n 64 -r 1 -o md"

run() {  # label devices split model
  local label=$1 dev=$2 sm=$3 model=$4
  echo "=== $label === idle $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr '\n' ' ')"
  CUDA_VISIBLE_DEVICES=$dev GGML_CUDA_ALLREDUCE=none nsys profile -t cuda,nvtx -s none --cpuctxsw=none \
    --force-overwrite true -o $OUT/$label $B -m $model -sm $sm $COMMON > $OUT/$label.log 2>&1 &
  local pid=$! peak=0
  while kill -0 $pid 2>/dev/null; do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
    (( t > peak )) && peak=$t
    if (( t >= 80 )); then echo "ABORT: $t C"; pkill -f llama-bench; kill $pid; wait $pid; exit 1; fi
    sleep 2
  done
  wait $pid; echo "exit $? | peak ${peak}C"; grep -E "^\|" $OUT/$label.log
}

run nsys-2card-q4km 0,1 tensor /root/Qwen3.8-27B-UD-Q4_K_M.gguf
run nsys-1card-iq3s 0 none /root/Qwen3.8-27B-UD-IQ3_S.gguf
