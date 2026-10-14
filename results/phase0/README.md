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

---

# Phase 0.2 — nsys trace, 2-card vs 1-card (2026-10-14)

`scripts/run-phase0-nsys.sh`: `llama-bench -p 2048 -n 64 -r 1` (with warmup), traced with nsys
2022.4 (`-t cuda,nvtx`). Arms: 2-card Q4_K_M `-sm tensor` with `GGML_CUDA_ALLREDUCE=none`, and
1-card IQ3_S `-sm none`. Peak temperatures were 57 °C and 60 °C. Analysis is `scripts/nsys_budget.py`;
output is in `nsys/*-budget.txt`. The `.nsys-rep` files open in the Nsight Systems GUI. Regenerate the
SQLite with `nsys export --type sqlite`.

llama-bench under nsys measured:
- 2-card: pp2048 381 t/s, tg64 18.3 t/s.
- 1-card: pp2048 215 t/s, tg64 11.2 t/s.

Tracing adds ~2–4 ms per decode token. That overhead is included in the decode idle figures below.

| | 2-card pp2048 | 1-card pp2048 | 2-card tg (per token) | 1-card tg (per token) |
|---|---|---|---|---|
| Wall | 5,275 ms | 9,179 ms | 58.7 ms | 91.9 ms |
| GPU kernel busy | **82%** | 100% | **77%** | 96% |
| GPU idle | **948 ms (18%)** | 3 ms | **13.6 ms (23%)** | 3.9 ms (4%) |
| Memcpy (per card) | 919 ms, 10.6 GB | 0 | 0.9 ms, 7 MB | 0.1 ms |
| GEMM (cuBLAS `maxwell_hgemm`) | 64% | 80% | — | — |
| Matvec (`mul_mat_vec_q`) | — | — | 63.5% (37.3 ms) | 86% (79 ms) |
| Dequant / convert | 5.8% | 7.3% | 3.2% | 2.2% |
| GDN scan | 4.7% | 6.3% | 0.9% | 1.3% |
| Flash attention (2k ctx) | 1.9% | 2.2% | 0.6% | 0.4% |
| AllReduce add | 1.5% | — | 1.9% | — |
| Kernels | 3.35k / ubatch | 3.2k / ubatch | ~2.2k / token | ~2.1k / token |

## What it means

- **2-card prefill: communication is serialised and costs ~18%.**
  - Each card copies 10.6 GB per ubatch in 919 ms (~11.5 GB/s). The GPU sits idle for that whole time: kernel busy time plus memcpy time equals wall time.
  - That clears P1's gate (comms over 5%). Halving the bytes (P1) would save ~9%; overlapping copies with compute (P2) could recover most of the remainder.
- **2-card decode: ~23% idle, but it is not copy time.** Memcpy is under 1 ms per token. The ~13.6 ms per token of gaps (~10 ms once trace overhead is removed) is host-side sync and launch latency across ~2.2k kernels plus ~270 copies per token.
  - That clears D1's gate (launch gaps over 10%) and D3's gate.
- **1-card decode is 96% busy, and 86% of it is matvec.** CUDA graphs can gain at most ~4% there. D2, the matvec kernel, is the single-card lever.
- **The dequant-to-F16 pass before each prefill GEMM is 6–7%.** That is P3's "fuse dequantize into the GEMM tile load" target, in addition to GEMM efficiency itself.
- The 0.1 estimates held: the op sums predicted ~5.6 s for 2-card pp (measured 5.3 s at 2k ctx) and ~1 s of comms (measured 0.95 s).
