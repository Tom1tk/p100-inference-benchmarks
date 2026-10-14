# Phase 0 — op-level time budget (one P100, 2026-10-14)

Fork [Tom1tk/llama.cpp-Pascal](https://github.com/Tom1tk/llama.cpp-Pascal) at `master`,
`build-cuda-p100` (sm_60). One card (GPU0), stock clocks, 175 W cap, peak 60 °C.

## Method

1. `test-export-graph-ops` (CPU only) dumped every op shape of the real graph,
   `-c 4096 -b 2048 -ub 2048 -fa on`, to `ops-UD-{Q4_K_M,IQ3_S}.txt`.
2. `scripts/run-opperf.sh` ran `test-backend-ops perf -b CUDA0 --test-file ... --output sql`
   with a live 80 °C kill switch → `opperf-*.sql`.
3. `scripts/budget.py` maps each GGUF tensor (type, shape) to its measured op time
   and sums per token (decode, n=1) and per 2048-token ubatch (prefill). MTP block
   excluded. **Not counted:** lm_head, norms, elementwise ops, launch gaps — so these
   are lower bounds on real wall time, and the split is of *measured op* time.

## Kernel efficiency

| Path | Measured | Reference | Efficiency |
|---|---|---|---|
| Decode matvec, K-quants / IQ4 / q8_0 | 180–235 GB/s | 493 GB/s (plain copy), 732 HBM peak | ~40% of achievable |
| Decode matvec, IQ3_S / IQ2 / IQ3_XXS / q3_K / q2_K | 85–115 GB/s | same | ~20% of achievable |
| Prefill GEMM, quantized weights, n=2048 | 12–13 TFLOPS | 18.7 FP16 peak | ~65–70% |

Decode matvec is **ALU-bound, not bandwidth-bound** — consistent with the emulated
`dp4a` path on sm_60.

## Time budget (one card)

| | Q4_K_M | IQ3_S |
|---|---|---|
| **Decode, ms/token (op sum)** | 69.6 | 77.6 |
| FFN matvec | 66% | 68% |
| GDN projections | 24% | 24% |
| Full-attention projections | 7% | 7% |
| GDN scan | 1.8% | 1.6% |
| **Prefill, ms per 2048 ubatch (op sum)** | 9,173 | 9,288 |
| GEMM total (FFN 62 / GDN proj 19 / attn 6) | ~87% | ~87% |
| Flash attention (4k ctx) | 6.6% | 6.6% |
| GDN scan | 5.9% | 5.9% |
| SSM conv | 0.4% | 0.4% |

Raw: `budget-1card.txt`.

## What it means

- **GEMMs are ~85% of prefill op time**, not ~15%. This contradicts the H18 reading
  of H17 (see the fix note there). The GDN scan is ~6%.
- **Decode headroom is in matvec**: at ~450 GB/s, 2-card drafter-free decode would go
  from ~20 to ~34 t/s (estimate, not measured).
- **Two-card overhead (inferred)**: the pair measured 6.6 s per ubatch at 16k; half the
  1-card op sum with FA scaled to 16k is ~5.6 s, so roughly 1 s (~15%) is comms/sync.
  Pair decode is 49 ms/token against ~35 ms of matvec per card, so ~14 ms is
  comms/launch. Both to be confirmed by an nsys profile.
