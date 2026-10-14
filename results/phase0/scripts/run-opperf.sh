#!/bin/bash
# test-backend-ops perf on CUDA0 for the exported model ops, with a live 80 C kill switch.
set -u
BIN=/root/llama.cpp-Pascal/build-cuda-p100/bin/test-backend-ops
OUT=/root/p100-benchmarks/results/phase0
ABORT_TEMP=80
for m in UD-Q4_K_M UD-IQ3_S; do
  CUDA_VISIBLE_DEVICES=0 $BIN perf -b CUDA0 --test-file $OUT/ops-$m.txt --output sql > $OUT/opperf-$m.sql 2> $OUT/opperf-$m.log &
  PID=$!
  PEAK=0
  while kill -0 $PID 2>/dev/null; do
    T=$(nvidia-smi -i 0 --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    (( T > PEAK )) && PEAK=$T
    if (( T >= ABORT_TEMP )); then echo "ABORT: GPU0 at ${T}C"; kill $PID; exit 3; fi
    sleep 1
  done
  wait $PID; echo "$m rc=$? peak ${PEAK}C rows=$(grep -c INSERT $OUT/opperf-$m.sql)"
done
