# Dual Tesla P100 — Qwen3.8-27B Inference Engine Benchmarks

Benchmarking four llama.cpp-family inference engines on a dual Tesla P100
(sm_60) rig to find the fastest stable configuration for serving Qwen3.8-27B.

**If you are an agent picking this work up: read [RUNBOOK.md](RUNBOOK.md) first.**
It is the operating procedure — exact commands, safety limits, failure handling,
and the mandatory commit/push protocol. Everything else is reference.

---

## Current status

| | |
|---|---|
| Phase 0 — smoke tests | **Done** |
| Phase 1 — engine baseline | **Done** except the single-GPU (`none`) leg, deferred |
| Phase 2 — drafters on/off | **Done** |
| Phase 3 — quant sweep | Not started (a partial KV-quant sweep was run — see RESULTS) |
| Phase 4 — hypothesis tests | **9 of 12 settled** — see below |
| Phase 5 — agentic web build | Not started (tooling ready) |
| Phase 6 — dual-GPU transport (H10–H12) | H10 and H11 closed; H12 open but low value |

### Best measured configuration

```
mainline-rebased (57affa09)  ·  -sm tensor  ·  GGML_CUDA_ALLREDUCE=none
Qwen3.8-27B-UD-Q4_K_M  ·  --spec-type draft-mtp -md mtp-Qwen3.8-27B-Q4_0.gguf
-ngl 99 -fa 1 -t 8 -ctk f16 -ctv f16
```

**29.26 t/s decode · 198.4 t/s prefill at 16k context**, 73.3% draft acceptance.
For comparison, the best layer-split configuration reaches 19.05 t/s and the
Phase 1 drafter-free best was 22.05.

⚠️ `GGML_CUDA_P2P=1` measured +2.9% decode *on its own*, drafter-free. The
combination with MTP was never measured — **do not quote ~30 t/s.**

### What is settled

| | |
|---|---|
| Split mode | `-sm tensor` wins, and it matters more than the drafter (+55–60% either way) |
| Drafter | MTP. Its speedup is depth-dependent: 1.05× at 4k, 1.48× at 16k |
| Engine | `mainline-rebased`. `buun` trails 12–14% on prefill; `ik` has no tensor split |
| KV cache | `f16`. TurboQuant costs 14.3% of decode to save 788 MiB |
| DFlash2 | Layer-split-only, so it cannot be used. See H8 |
| AllReduce | `none`. `internal` needs Volta and silently falls back on Pascal |

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
