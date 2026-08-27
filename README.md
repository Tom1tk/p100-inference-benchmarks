# Dual Tesla P100 — Qwen3.8-27B Inference Engine Benchmarks

Benchmarking four llama.cpp-family inference engines on a dual Tesla P100
(sm_60) rig to find the fastest stable configuration for serving Qwen3.8-27B.

**If you are an agent picking this work up: read [RUNBOOK.md](RUNBOOK.md) first.**
It is the operating procedure — exact commands, safety limits, failure handling,
and the mandatory commit/push protocol. Everything else is reference.

**Two hard limits before you launch anything:** never let a card exceed 83 °C,
and remember that **every run costs the user real electricity**. Keep tests
short, never re-measure a settled lever, and abort a sweep once its conclusion
is clear — RUNBOOK §0 and §2.1.

---

## The objective

**Everything in this repo exists to produce one deliverable: a single server
launch command (or a small set of them) for serving Qwen3.8-27B on this rig,
primarily for coding work.** Not a table of benchmarks — a command line. Every
hypothesis should be read as "does this change what goes in that command."

Four targets, and all four must hold at once. That simultaneity is the whole
difficulty; almost every lever trades one against another:

| Target | Currently |
|---|---|
| **100k context** | **MET (H26).** Serves at 100k with `q8_0` KV and MTP, 12,147 MiB/card, 4,122 MiB/card spare |
| **Highest possible decode** | **20.02 t/s at 100k** (H26); 27.41 at 16k, 29.39 at 16k/`-ub 512` (H24) |
| **Highest possible prefill / TTFT** | **207.4 t/s / ~7.8 min TTFT at 100k** (H26); 310.5 t/s at 16k (H24) |
| **Output quality indistinguishable from baseline** | **Still not measured against anything.** The only target with zero data |

Three of the four now have numbers at the target depth, taken together in one
run rather than inferred from separate ones. The fourth is the constraint that
makes the other three hard, and it is the one with **no** data behind it today —
it is now the only thing standing between this repo and its deliverable. It is why PFlash and DFlash v1 are excluded,
why TurboQuant's 14.3% decode cost was judged not worth 788 MiB, and why any
throughput win from a lossy technique has to clear a NIAH gate before it counts.

**The levers, by category** — this is the search space the phases explore:

| For | Options |
|---|---|
| **Context** | KV-cache quantisation (`q8_0` ✓ H25 — 1.5% of decode, halves the cache); cache reuse |
| **Decode** | MTP, DFlash2, split mode, lower weight quantisation, P2P |
| **Prefill / TTFT** | ubatch/batch (H14 ✓, decode cost priced by H24 ✓), chunked GDN (H22 — crashes on `-sm tensor`), power cap (H15), cache reuse (H19) |
| **Quality** | Quantisation tiers and speculative-decoding settings, benchmarked against baseline |

**Deployment mitigation, decided 2026-08-25.** Long TTFT at 100k is partly a
*usage* problem, not only an engineering one. The intended real-world use is a
long-running autonomous harness — prime-agent, or anything with a `/goal` mode —
where a large prefill is amortised across a long session instead of being paid per
turn. This is why H19 (prompt-cache reuse) is rated the highest-value practical
item: it is the lever that matches how the rig will actually be used.

---

## Current status

| | |
|---|---|
| Phase 0 — smoke tests | **Done** |
| Phase 1 — engine baseline | **Done** except the single-GPU (`none`) leg, deferred |
| Phase 2 — drafters on/off | **Done** |
| Phase 3 — quant sweep | Not started (a partial KV-quant sweep was run — see RESULTS) |
| Phase 4 — hypothesis tests | **9 of 12 settled**; 9 new opened (H13–H21) |
| Phase 5 — agentic web build | Not started (tooling ready) |
| Phase 6 — dual-GPU transport (H10–H12) | H10 and H11 closed; H12 open but low value |
| **Phase 7 — prefill & TTFT (H13–H23)** | **Open.** H13 and H14 confirmed (100k measured: 215.4 t/s, 7.7 min TTFT); H16 and H21 withdrawn; H22 tested and crashes on `-sm tensor` (unwired split-axis metadata, not a speed verdict); H23 opened, may outrank H22 as the 100k-specific lever; **H26 served the target depth end to end for the first time (20.02 t/s decode, 207.4 prefill, 12,147 MiB/card, 70 C sustained)**; H24 priced `-ub 2048`'s decode cost at 6.7% and kept it; **H20/H25 closed 2026-08-27 — a single P100 is not feasible for real-world use** (56% of the pair's prefill, 38% of its real decode, MTP OOMs, cannot reach 100k), and H25 corrected the KV-quant penalty from 14.3% to 1.5%. See [Research/prefill-ttft-2026-08-25.md](Research/prefill-ttft-2026-08-25.md) and [Research/chunked-gdn-2026-08-25.md](Research/chunked-gdn-2026-08-25.md) |

### Best measured configuration

```
mainline-rebased (57affa09)  ·  -sm tensor  ·  GGML_CUDA_ALLREDUCE=none
Qwen3.8-27B-UD-Q4_K_M  ·  --spec-type draft-mtp -md mtp-Qwen3.8-27B-Q4_0.gguf
-ngl 99 -fa 1 -t 8 -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 -c 100000 -np 1
```

**Measured end to end at 100k (H26, 2026-08-27): 20.02 t/s decode · 207.4 t/s
prefill**, 60.3% draft acceptance, 12,147 MiB/card, 70 C peak, ~7.8 min TTFT.
This is the first time the target depth has been served rather than simulated,
and it leaves **4,122 MiB/card spare** — `q4_0` KV is not needed.

Same config at 16k (H24): 27.41 t/s decode · 310.5 t/s prefill, 66.4%
acceptance, 10,973 MiB/card. Going from 16k to 100k costs **27% of decode and
33% of prefill**, and drops acceptance 6 points.

Still unmeasured at any depth: **quality**. Everything above is speed.

`-ub 2048` **costs 6.7% of decode** against the old `-ub 512` (29.39 t/s /
198.5 t/s / 73.3%) and buys **+56.4% of prefill**. The decode loss is not the
decode path — it is entirely a 6.9 pp drop in drafter acceptance, and the
generated tokens are byte-identical between the two settings. At 16k the trade
saves 29 s per turn and breaks even only at ~11,800 output tokens, so `-ub 2048`
stays. A **drafter-free** config should get it for free. See H24.

For comparison, the best layer-split configuration reaches 19.05 t/s and the
Phase 1 drafter-free best was 22.05. H14's +63% prefill at 2–8k **largely
evaporates by 100k** (H13) — see below.

⚠️ `GGML_CUDA_P2P=1` measured +2.9% decode *on its own*, drafter-free. The
combination with MTP was never measured — **do not quote ~30 t/s.**

### What is settled

| | |
|---|---|
| Split mode | `-sm tensor` wins, and it matters more than the drafter (+55–60% either way) |
| Drafter | MTP. Its speedup is depth-dependent: 1.05× at 4k, 1.48× at 16k |
| Engine | `mainline-rebased`. `buun` trails 12–14% on prefill; `ik` has no tensor split |
| KV cache | **`q8_0`** — costs 1.5% of decode and halves the cache (H25). The 14.3% figure is **TurboQuant's**, not KV quantisation's, and was wrongly generalised |
| DFlash2 | Layer-split-only, so it cannot be used. See H8 |
| AllReduce | `none`. `internal` needs Volta and silently falls back on Pascal |

### TTFT at 64k–100k — now measured, not projected

**Updated 2026-08-26 — H13, real numbers at `-ub 2048`.** 1743 s run, peak
70 °C, `results/raw/h13-prefill-depth-q4km.csv`:

| | 16k | 64k | 100k |
|---|---|---|---|
| prefill t/s, `-ub 2048` | **327.1** | **250.1** | **215.4** |
| TTFT, `-ub 2048` | **50 s** | **4.4 min** | **7.7 min** |

The 8k measurement's +63% gain (H14) does **not** hold at depth: prefill falls
34% from 16k to 100k at the same settings. The 100k TTFT (7.7 min) lands close
to the *best case* of the old `-ub 512` bracket (7.8–15.9 min) — most of H14's
win is gone by the time it matters for the actual target.

Three facts that bound every proposed fix:

- **Hardware floor — revise downward.** The standing "54 GFLOP/token ×
  100k ÷ 37.4 TFLOPS = 144 s minimum" figure is the linear `2×params`
  approximation, and it is **wrong at 100k**: it treats attention as
  negligible, true at the 2–8k depths it was fit against, false once
  n (100,000) ≫ d_model (5120). A quadratic term this large changes what
  "% of peak" and "tuning budget left" even mean at depth — see H23. Don't
  quote the 52%/30%-of-peak figures for 100k; they were computed against the
  wrong FLOP count.
- **The model is a hybrid, and layer-count share is not FLOP-count share.**
  `full_attention_interval = 4` — only ~16 of 65 layers are attention, which
  is why sparse attention was written off as low-ceiling. **That reasoning
  is now suspect at 100k specifically** (H23): those 16 layers' O(n²) cost
  may dominate total prefill FLOPs at depth even though they're a minority of
  layers. Back-of-envelope, not measured — see H23 before acting on it.
- **The gated-delta-net kernel is still a real, separate suspect for the
  short-context number, but is probably not what's causing the depth decay.**
  Its grid contains no `n_tokens` and its cost is linear in tokens — a linear
  kernel cannot on its own produce the *falling* t/s H13 just measured; only
  a super-linear (quadratic) term can. H18 (GDN-bound) and H23
  (attention-bound) are likely both true at different depths — GDN matters
  more at 2–16k, attention more at 64k–100k. **The chunked GDN algorithm
  already exists as a ggml graph and runs on sm_60** — still worth sizing
  (H18) before building (H22), but H23 may be the bigger 100k-specific prize.
  See [Research/chunked-gdn-2026-08-25.md](Research/chunked-gdn-2026-08-25.md).

Also newly known and not yet acted on: **the cards are power-capped at 175 W of a
250 W default** — raising it is now **approved to a 220 W ceiling** (H15), blocked
only on an in-person PSU plug-meter check — and **our builds carry a live sm_60
arithmetic bug** that flips ~1 in 20 greedy tokens, which `buun` has already
fixed, so cross-engine *quality* comparisons made in this repo are confounded (H17).

**PFlash is closed outright (2026-08-25)** — product, fork and technique. It needs
sm_80, there is no v2, and hand-porting the selection stage (H21) is not worth the
build cost. Its earlier "too lossy" verdict was reached on a **7900 XTX**, not on
these cards, so it neither confounds nor is confounded by anything here.

**Two measurement rules**, both learned by getting them wrong: never size a
speculative-decoding run at 64 tokens (it overstated a speedup by 13x), and
never compare acceptance rates across engines without pinning `n_max`/`p_min`.

**Resolved (2026-08-24):** DFlash2 runs — upstream PR #27342 has v2 and builds
for `sm_60`. It is nonetheless unusable in the winning configuration (H8).

**Resolved (2026-08-24):** the `ik_llama` `-sm graph` NCCL abort. NCCL is broken
rig-wide, not just on `ik`; on mainline it is switchable at runtime. See H5.

**Validated:** cooling is adequate — the hottest cell of the whole project
peaked at 70°C against an 83°C throttle. Power cap (175W/card) and persistence
mode are confirmed applied.

---

## Documents

| File | What's in it |
|---|---|
| [RUNBOOK.md](RUNBOOK.md) | **Start here.** How to run a benchmark, safety limits, failure handling, git protocol, phase completion criteria |
| [METHODOLOGY.md](METHODOLOGY.md) | Hardware, engines, models, fixed parameters, and why each was chosen |
| [HYPOTHESES.md](HYPOTHESES.md) | H1–H24 — the open questions each run is meant to answer, with current status |
| `Research/` | Deep-dive research documents, distilled into the hypotheses above |
| [WEB_BENCH.md](WEB_BENCH.md) | Phase 5 — the agentic web-build benchmark: port scheme, metrics, quality scoring |
| [RESULTS.md](RESULTS.md) | Curated result tables, one section per phase |
| [RUNLOG.md](RUNLOG.md) | Chronological log — what ran, what broke, what changed |
| `results/all-results.csv` | Machine-readable aggregate of every run |
| `logs/` | Raw stdout/stderr and GPU telemetry per run |
| `results/web-bench.csv` | Machine-readable aggregate of every Phase 5 run |
| `sites/` | Websites the models built in Phase 5 — quality evidence |
| `scripts/` | `run-bench.sh` (Phases 1–4), `run-web-bench.sh` (Phase 5), `web_bench_metrics.py` (per-request timings), `gpu-monitor.sh` (telemetry) |

---

## Quick start

```bash
cd /root/p100-benchmarks
./scripts/run-bench.sh pflash /root/Qwen3.8-27B-UD-Q6_K_M.gguf layer phase1-pflash-layer-q6k
```

Phase 5, the agentic web-build benchmark:

```bash
./scripts/run-web-bench.sh buun /root/Qwen3.8-27B-UD-Q6_K_M.gguf layer p5-buun-layer-q6k 0
```

Both scripts handle preflight checks, telemetry, logging, and the commit/push.
Read [RUNBOOK.md](RUNBOOK.md) before running either — there are real gotchas
(model loads take 4–8 minutes and look like hangs). For Phase 5 also read
[WEB_BENCH.md](WEB_BENCH.md): every run claims an index that fixes its ports,
and indices must not be reused.

---

## Prior knowledge

`/root/p100_inference_knowledge_handover.md` is a knowledge-transfer document
from earlier work on this rig. Its hardware facts are reliable; its
**performance claims predate the current engine builds and are treated as
hypotheses to verify, not settled findings** — that's what H1–H7 are.

Two of its claims have already been re-tested here: the NCCL warning (H5)
proved *more* severe than documented, and its `-sm` support table for
`ik_llama` was wrong. Verify, don't assume.
