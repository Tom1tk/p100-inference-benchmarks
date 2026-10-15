# Dual Tesla P100 — Qwen3.8-27B serving benchmarks

Finding the fastest stable way to serve Qwen3.8-27B on a dual Tesla P100
(sm_60) rig, primarily for coding work.

**Agents: read [RUNBOOK.md](RUNBOOK.md) before running anything.** It has the
hard limits (80 °C abort, electricity), the procedure, and the work queue.

---

## The objective

One deliverable: **a single `llama-server` launch command** that hits all four
targets at once. Every hypothesis is read as "does this change what goes in
that command."

| Target | Status |
|---|---|
| 100k context | **Met** (H26) — 12,147 MiB/card, 4,122 MiB spare |
| Highest decode | **46.4 t/s at 16k** (fork command, `h45-head96k-16k`); **28.4 t/s at 98k** (`h46-fagemm-100k`, +42% vs 20.02 on the old stock command, H26) |
| Highest prefill / TTFT | **276 t/s, ~355 s TTFT at 98k** (`h46-fagemm-100k`; +33% vs 207 stock, H26); 365 t/s at 16k |
| Output quality indistinguishable from baseline | **Met** (H32): 19/19 on the 100k haystack vs 18/19 for the Q5_K_M FP32 reference; KLD 0.057 mean, mostly the quant step (H30) |

Quality is the binding constraint: it is why PFlash and DFlash v1 are excluded,
why TurboQuant's 14.3% decode cost was refused, and why any lossy speed win
must clear a NIAH gate before it counts.

The intended use is a long-running autonomous harness, where one large prefill
is amortised over a session. That makes prompt-cache reuse (H19) the highest
value practical lever left.

## The current answer

```
Tom1tk/llama.cpp-Pascal (fork, /root/llama.cpp-Pascal/build-cuda-p100)  ·  -sm tensor
GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 GGML_CUDA_PASCAL_MMVH=1 GGML_PASCAL_F16_REDUCE=1 GGML_CUDA_PASCAL_GEMM_ALGO=1 GGML_PASCAL_DUPLEX_REDUCE=1 GGML_PASCAL_MMVH_Q6K_WIDE=1 GGML_PASCAL_MMVH_LOWBIT=1 GGML_PASCAL_FA_GEMM=1
Qwen3.8-27B-UD-Q4_K_M  ·  --spec-type draft-mtp -md mtp-Qwen3.8-27B-Q4_0-head96k.gguf
-ngl 99 -fa 1 -t 8 -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 -c 100000 -np 1
```

Adopted 2026-10-14 after it passed the Phase 8 quality gate (H32: 19/19 at 100k, vs 18/19
for the Q5_K_M FP32 reference). Fork levers: half2 matvec incl. Q6_K (H28, H34), F16
AllReduce payload (H31/H34), cuBLAS ALGO3 for n ≥ 512 (H33, H35), full-duplex AllReduce copies (H36, bit-identical, so H32 still holds), 4-lane Q6_K matvec (H41b; fp32 reorder only, ~1e-6 relative, PPL not worse; H32 not re-run), branch-free Q4_K/Q5_K scale unpack (H43, output identical, no flag), Q3_K + IQ4_NL on mmvh (H44, `LOWBIT`, output identical, KLD gate passed), drafter LM head trimmed to the first 98,304 vocab rows (H45; lossless by construction since the target verifies every draft, output identical, acceptance unchanged), prefill flash attention through cuBLAS GEMMs at depth (H46, `FA_GEMM`: +19.2% prefill at 98k; lossy, so the H32 haystack was re-run: 19/19). The 16k row predates H46.

The trimmed drafter is generated, not downloaded: `python3 /root/llama.cpp-Pascal/scripts/pascal-trim-mtp-head.py /root/mtp-Qwen3.8-27B-Q4_0.gguf /root/mtp-Qwen3.8-27B-Q4_0-head96k.gguf 98304` (needs fork cfea757+fd90268; stock builds reject the shorter head).

| Depth | Decode | Prefill | Acceptance | VRAM/card | Peak | Run |
|---|---|---|---|---|---|---|
| 98k | 28.38 t/s | 275.75 t/s (~355 s TTFT) | 60.5% | 12,331 MiB | 67 °C | `h46-fagemm-100k` (q8_0 KV, REPS=3; control `h46-ctrl-100k` 29.65 / 231.36 / 65.3%; decode dip is the changed greedy text drafting worse, not the kernel, which only runs for n_q ≥ 512) |
| 16k | 46.35 t/s | 365.8 t/s | 72.8% | 11,157 MiB | 65 °C | `h45-head96k-16k` (f16 KV, REPS=3; +3.1% decode from the 96k-row drafter head, fork fd90268, vs `h44-lowbit-16k` 44.96 / 366.4) |

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
| 8 quality gate | Tier 1 (H30 KLD) and tier 2 (H32 100k haystack) passed ([QUALITY-PLAN.md](QUALITY-PLAN.md)) |

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
SPLIT=tensor GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 CTX=16384 UB=2048 BATCH=2048 \
  ./scripts/run-spec-placement.sh <label> draft-mtp /root/mtp-Qwen3.8-27B-Q4_0.gguf default
```

Both handle preflight, telemetry, logging and the commit/push. Model loads take
4–8 minutes with no output; that is not a hang.

## Prior knowledge

`/root/p100_inference_knowledge_handover.md` predates these builds. Its hardware
facts are reliable; its performance claims were treated as hypotheses (H1–H7)
and several proved wrong. Verify, don't assume.
