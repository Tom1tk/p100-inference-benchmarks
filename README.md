# Dual Tesla P100 — Qwen3.8-27B Inference Engine Benchmarks

Benchmarking three llama.cpp-family inference engines on a dual Tesla P100
(sm_60) rig to find the fastest stable configuration for serving Qwen3.8-27B.

**If you are an agent picking this work up: read [RUNBOOK.md](RUNBOOK.md) first.**
It is the operating procedure — exact commands, safety limits, failure handling,
and the mandatory commit/push protocol. Everything else is reference.

---

## Current status

| | |
|---|---|
| Phase 0 — smoke tests | **Done** (cooling validated, one engine blocked) |
| Phase 1 — engine baseline | **In progress** (pflash/layer done) |
| Phase 2 — drafters on/off | Not started |
| Phase 3 — quant sweep | Not started |
| Phase 4 — hypothesis tests | Not started |
| Phase 5 — agentic web build | Not started (tooling ready) |
| Phase 6 — dual-GPU transport (H10–H12) | Not started |

**Resolved (2026-08-24):** DFlash2 now runs. The three forks all have DFlash
**v1**; upstream `ggml-org/llama.cpp` **PR #27342** has v2, and it builds for
`sm_60`. Added as a fourth engine, `mainline`. Both drafters are on disk and
metadata-verified as genuine v2. See H8.

**Blocker 1:** `ik_llama` `-sm graph` aborts with `ncclAllReduce failed with
status 1`. Fix is known and untried — rebuild with `-DGGML_NCCL=OFF`. See
[RUNLOG.md](RUNLOG.md) and H5 in [HYPOTHESES.md](HYPOTHESES.md).

**Validated:** cooling is adequate — a full prefill sweep peaked at 63°C
against an 83°C throttle. Power cap (175W/card) and persistence mode are
confirmed applied.

---

## Documents

| File | What's in it |
|---|---|
| [RUNBOOK.md](RUNBOOK.md) | **Start here.** How to run a benchmark, safety limits, failure handling, git protocol, phase completion criteria |
| [METHODOLOGY.md](METHODOLOGY.md) | Hardware, engines, models, fixed parameters, and why each was chosen |
| [HYPOTHESES.md](HYPOTHESES.md) | H1–H12 — the open questions each run is meant to answer, with current status |
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
