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
| Phase 1 — engine baseline | **Not started** |
| Phase 2 — MTP on/off | Not started |
| Phase 3 — quant sweep | Not started |
| Phase 4 — hypothesis tests | Not started |

**Blocker:** `ik_llama` `-sm graph` aborts with `ncclAllReduce failed with
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
| [HYPOTHESES.md](HYPOTHESES.md) | H1–H7 — the open questions each run is meant to answer, with current status |
| [RESULTS.md](RESULTS.md) | Curated result tables, one section per phase |
| [RUNLOG.md](RUNLOG.md) | Chronological log — what ran, what broke, what changed |
| `results/all-results.csv` | Machine-readable aggregate of every run |
| `logs/` | Raw stdout/stderr and GPU telemetry per run |
| `scripts/` | `run-bench.sh` (run + log + commit + push), `gpu-monitor.sh` (telemetry) |

---

## Quick start

```bash
cd /root/p100-benchmarks
./scripts/run-bench.sh pflash /root/Qwen3.8-27B-UD-Q6_K_M.gguf layer phase1-pflash-layer-q6k
```

The script handles preflight checks, telemetry, logging, and the commit/push.
Read [RUNBOOK.md](RUNBOOK.md) before running it — there are real gotchas
(model loads take 4–8 minutes and look like hangs).

---

## Prior knowledge

`/root/p100_inference_knowledge_handover.md` is a knowledge-transfer document
from earlier work on this rig. Its hardware facts are reliable; its
**performance claims predate the current engine builds and are treated as
hypotheses to verify, not settled findings** — that's what H1–H7 are.

Two of its claims have already been re-tested here: the NCCL warning (H5)
proved *more* severe than documented, and its `-sm` support table for
`ik_llama` was wrong. Verify, don't assume.
