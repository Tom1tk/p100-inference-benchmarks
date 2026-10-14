#!/bin/bash
# Fork P2 step 1: prefill idle left with the full adopted set (adds H36 duplex reduce, H41b Q6_K wide)?
# Same trace as Phase 0.2 (llama-bench pp2048 + tg64, r=1). 2-card only: 1 card has no reduce
# to overlap, so a 1-card arm would re-measure Phase 0.2 (rule zero). Gate: P2 is built only if
# prefill idle is still > 5% of wall.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
B=/root/llama.cpp-Pascal/build-cuda-p100/bin/llama-bench
OUT=results/p2-step1; mkdir -p $OUT
L=nsys-2card-q4km-serve-set
echo "=== $L === idle $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr '\n' ' ')"
CUDA_VISIBLE_DEVICES=0,1 GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_MMVH=1 \
GGML_PASCAL_F16_REDUCE=1 GGML_CUDA_PASCAL_GEMM_ALGO=1 GGML_PASCAL_DUPLEX_REDUCE=1 GGML_PASCAL_MMVH_Q6K_WIDE=1 \
  nsys profile -t cuda,nvtx -s none --cpuctxsw=none --force-overwrite true -o $OUT/$L \
  $B -m /root/Qwen3.8-27B-UD-Q4_K_M.gguf -sm tensor -ngl 99 -fa 1 -t 8 -ctk f16 -ctv f16 \
  -b 2048 -ub 2048 -p 2048 -n 64 -r 1 -o md > $OUT/$L.log 2>&1 &
pid=$! peak=0
while kill -0 $pid 2>/dev/null; do
  t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
  (( t > peak )) && peak=$t
  if (( t >= 80 )); then echo "ABORT: $t C"; pkill -f llama-bench; kill $pid; wait $pid; exit 1; fi
  sleep 2
done
wait $pid; echo "exit $? | peak ${peak}C"; grep -E "^\|" $OUT/$L.log
/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter -i $OUT/$L.qdstrm >/dev/null  # nsys 2022.4 deb skips the import step
nsys export --type sqlite --force-overwrite true -o $OUT/$L.sqlite $OUT/$L.nsys-rep >/dev/null
python3 results/phase0/scripts/nsys_budget.py $OUT/$L.sqlite | tee $OUT/$L-budget.txt
git add $OUT/*.log $OUT/*-budget.txt $OUT/*.nsys-rep .gitignore scripts/run-p2-step1-nsys.sh
git commit -q -m "bench: fork P2 step 1 — nsys budget with the full serve set" && { git push -q && echo pushed; }
