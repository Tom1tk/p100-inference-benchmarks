#!/usr/bin/env bash
# H55: find the op behind the n=4 step cliff (verify batch 4 -> 5 columns). test-backend-ops perf on CUDA0, serve env,
# REPS=3: MUL_MAT per weight type at per-card shapes, n=4 vs 5; FLASH_ATTN_EXT f16 nb 4 vs 5. No server.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 GGML_CUDA_PASCAL_GEMM_ALGO=1 GGML_PASCAL_FA_GEMM=1 \
       GGML_PASCAL_MMVH_Q6K_WIDE=1 GGML_PASCAL_MMVH_LOWBIT=1
TBO=/root/llama.cpp-Pascal/build-cuda-p100/bin/test-backend-ops
OUT=results/h55-ncols.txt
for _ in $(seq 1 30); do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
    [[ "$t" -le 55 ]] && break; echo "cooling: hottest card ${t}C"; sleep 20
done
for rep in 1 2 3; do
    echo "## rep $rep"
    $TBO perf -b CUDA0 -o MUL_MAT -p 'type_b=f32,m=(8704|5120|124160),n=(4|5),k=(5120|8704|3072),bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\],k_v=0,o=1' 2>&1 | grep -E 'MUL_MAT\('
    $TBO perf -b CUDA0 -o FLASH_ATTN_EXT -p 'hsk=256,hsv=256,nh=2,nr23=\[6,1\],kv=(16384|98304),nb=(4|5),.*type_K=f16.*n_kv_max=0' 2>&1 | grep -E 'FLASH_ATTN_EXT\('
done | tee "$OUT"
echo "## mean us/run over reps (shape, n4, n5, delta)" | tee -a "$OUT"
grep -E 'MUL_MAT|FLASH' "$OUT" | grep -v '^##' | python3 -c '
import sys, re, collections
d = collections.defaultdict(list)
for l in sys.stdin:
    m = re.search(r"(MUL_MAT|FLASH_ATTN_EXT)\((.*)\):.* ([0-9.]+) us/run", l)
    if not m: continue
    v = m.group(2)
    n = re.search(r"\bn=(\d+)", v) or re.search(r"\bnb=(\d+)", v)
    key = re.sub(r"\b(n|nb)=\d+", r"\1=*", v)
    d[(m.group(1), key)].append((int(n.group(1)), float(m.group(3))))
for (op, key), xs in sorted(d.items()):
    mean = lambda c: sum(t for n, t in xs if n == c) / max(1, sum(1 for n, t in xs if n == c))
    a, b = mean(4), mean(5)
    short = re.sub(r"(type_b=f32,|,bs=.*|,mask=.*)", "", key)
    print(f"{op:15s} {short[:60]:60s} {a:9.1f} {b:9.1f} {b-a:+9.1f} ({100*(b/a-1) if a else 0:+.0f}%)")
' | tee -a "$OUT"
git add "$OUT" scripts/run-h55-ncols.sh
git commit -q -m "bench: H55 4 vs 5 column micro-bench" && { git push -q && echo pushed || echo "WARNING: push failed" >&2; }
