# Dual Tesla P100 — Qwen3.8-27B serving benchmarks

Finding the fastest stable way to serve Qwen3.8-27B on a dual Tesla P100
(sm_60) rig, primarily for coding work.

**Agents: read [RUNBOOK.md](RUNBOOK.md) before running anything.** It has the
hard limits (83 °C, electricity), the procedure, and the work queue.

---

## The objective

One deliverable: **a single `llama-server` launch command** that hits all four
targets at once. Every hypothesis is read as "does this change what goes in
that command."

| Target | Status |
|---|---|
| 100k context | **Met** (H26) — 12,147 MiB/card, 4,122 MiB spare |
| Highest decode | **20.02 t/s at 100k**, 27.41 at 16k (H26, H24) |
| Highest prefill / TTFT | **207.4 t/s, ~7.8 min TTFT at 100k**; 310.5 t/s at 16k |
| Output quality indistinguishable from baseline | **No data.** Phase 8, [QUALITY-PLAN.md](QUALITY-PLAN.md), scheduled last on purpose |

Quality is the binding constraint: it is why PFlash and DFlash v1 are excluded,
why TurboQuant's 14.3% decode cost was refused, and why any lossy speed win
must clear a NIAH gate before it counts.

The intended use is a long-running autonomous harness, where one large prefill
is amortised over a session. That makes prompt-cache reuse (H19) the highest
value practical lever left.

## The current answer

```
mainline-rebased (57affa09)  ·  -sm tensor  ·  GGML_CUDA_ALLREDUCE=none
Qwen3.8-27B-UD-Q4_K_M  ·  --spec-type draft-mtp -md mtp-Qwen3.8-27B-Q4_0.gguf
-ngl 99 -fa 1 -t 8 -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 -c 100000 -np 1
```

| Depth | Decode | Prefill | Acceptance | VRAM/card | Peak | Run |
|---|---|---|---|---|---|---|
| 100k | 20.02 t/s | 207.4 t/s | 60.3% | 12,147 MiB | 70 °C | `h26-100k-mtp-kvq8` |
| 16k | 27.41 t/s | 310.5 t/s | 66.4% | 10,973 MiB | 68 °C | `h24-ub2048-mtp-16k` |

Every lever behind that line, and every number, is in
[RESULTS.md](RESULTS.md). What is still open is in RUNBOOK §7.

## Status

| Phase | State |
|---|---|
| 0 smoke · 1 engine baseline · 2 drafters · 6 transport/placement | Done |
| 3 quant sweep | Folded into Phase 8 |
| 4 hypothesis tests | Ongoing — status table in [HYPOTHESES.md](HYPOTHESES.md) |
| 5 agentic web build | Not started, tooling ready ([WEB_BENCH.md](WEB_BENCH.md)) |
| 7 prefill / TTFT / 100k | Measured end to end (H13, H14, H24, H26). H15 (power cap) and H19 (cache reuse) remain |
| 8 quality gate | Planned, nothing run ([QUALITY-PLAN.md](QUALITY-PLAN.md)) |

## Documents

| File | What's in it |
|---|---|
| [RUNBOOK.md](RUNBOOK.md) | **Start here.** Procedure, safety, failure modes, git protocol, **work queue** (§7: open / parked / closed) |
| [RESULTS.md](RESULTS.md) | **All the data.** Table A (`llama-bench`), Table B (`llama-server`), Table C (lever verdicts) |
| [METHODOLOGY.md](METHODOLOGY.md) | Hardware, engines, models, fixed parameters, flag syntax per engine |
| [HYPOTHESES.md](HYPOTHESES.md) | H1–H26: claim, reasoning, test, result. The narrative behind every verdict |
| [RUNLOG.md](RUNLOG.md) | Chronological: what ran, what broke, what changed |
| [WEB_BENCH.md](WEB_BENCH.md) | Phase 5 procedure: port scheme, metrics, quality scoring |
| [QUALITY-PLAN.md](QUALITY-PLAN.md) | Phase 8 plan: quant sweep, drafter equivalence check, abort rules |
| `Research/` | Deep dives, distilled into the hypotheses |
| `results/raw/*.csv` | Per-run `llama-bench` output — the source for Table A (`scripts/build-matrix.py`) |
| `results/h11-placement.csv` | Every `llama-server` run — the source for Table B |
| `logs/` | stdout/stderr and GPU telemetry per run |
| `scripts/` | `run-bench.sh` (llama-bench), `run-spec-placement.sh` (llama-server), `run-web-bench.sh` (Phase 5), `gpu-monitor.sh` |

## Quick start

```bash
cd /root/p100-benchmarks
# llama-bench cell
./scripts/run-bench.sh mainline /root/Qwen3.8-27B-UD-Q4_K_M.gguf tensor <label>
# llama-server cell (drafter, acceptance, VRAM) — env knobs in the script header
SPLIT=tensor GGML_CUDA_ALLREDUCE=none CTX=16384 UB=2048 BATCH=2048 \
  ./scripts/run-spec-placement.sh <label> draft-mtp /root/mtp-Qwen3.8-27B-Q4_0.gguf default
```

Both handle preflight, telemetry, logging and the commit/push. Model loads take
4–8 minutes with no output; that is not a hang.

## Prior knowledge

`/root/p100_inference_knowledge_handover.md` predates these builds. Its hardware
facts are reliable; its performance claims were treated as hypotheses (H1–H7)
and several proved wrong. Verify, don't assume.
