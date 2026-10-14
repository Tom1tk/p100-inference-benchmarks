#!/bin/bash
# H40 follow-up: decode-only nsys trace (tg32) with GGML_PASCAL_PAR_DISPATCH=1, to see whether the two
# dispatch threads overlap their launches or serialise in the driver. One short run, CUDA API + kernels.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
B=/root/llama.cpp-Pascal/build-cuda-p100/bin/llama-bench
OUT=results/h40-trace; L=nsys-2card-tg32-par
CUDA_VISIBLE_DEVICES=0,1 GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 \
GGML_CUDA_PASCAL_GEMM_ALGO=1 GGML_PASCAL_DUPLEX_REDUCE=1 GGML_PASCAL_PAR_DISPATCH=1 \
  nsys profile -t cuda -s none --cpuctxsw=none --force-overwrite true -o $OUT/$L \
  $B -m /root/Qwen3.8-27B-UD-Q4_K_M.gguf -sm tensor -ngl 99 -fa 1 -t 8 -p 0 -n 32 -r 1 -o md > $OUT/$L.log 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
  t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
  if (( t >= 80 )); then echo "ABORT: $t C"; kill $pid; wait $pid; exit 1; fi
  sleep 2
done
wait $pid; echo "exit $?"; grep -E "^\|" $OUT/$L.log
/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter -i $OUT/$L.qdstrm >/dev/null
nsys export --type sqlite --force-overwrite true -o $OUT/$L.sqlite $OUT/$L.nsys-rep >/dev/null
rm -f $OUT/$L.qdstrm
