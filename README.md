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
| Highest decode | **47.5 t/s at 16k, 33.1 t/s at 100k** (final report, `r10s-fork-*`, 200 W): +65.6% and +58.5% vs the stock build (28.7 / 20.9) |
| Highest prefill / TTFT | **288 t/s, ~340 s TTFT at 100k; 392 t/s at 16k** (final report, 200 W): +34.4% and +23.2% vs stock (214 / 318) |
| Output quality indistinguishable from baseline | **Met**: 19/19 on the 100k haystack with the final f16-KV command (`p8-final-f16`; H32 19/19 with q8_0) vs 18/19 for the Q5_K_M FP32 reference; KLD 0.057 mean, mostly the quant step (H30) |

Quality is the binding constraint: it is why PFlash and DFlash v1 are excluded,
why TurboQuant's 14.3% decode cost was refused, and why any lossy speed win
must clear a NIAH gate before it counts.

The intended use is a long-running autonomous harness, where one large prefill
is amortised over a session. That makes prompt-cache reuse (H19) the highest
value practical lever left.

## The current answer

```
Tom1tk/llama.cpp-Pascal (fork, /root/llama.cpp-Pascal/build-cuda-p100)  ·  -sm tensor
GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1   (fork ≥ 5542bf9: the Pascal levers are on by default on sm_60)
Qwen3.8-27B-UD-Q4_K_M  ·  --spec-type draft-mtp -md mtp-Qwen3.8-27B-Q4_0-head96k.gguf
-ngl 99 -fa 1 -t 8 -ctk f16 -ctv f16 -b 2048 -ub 2048 -c 100000 -np 1
Power cap: 200 W per card (user-set 2026-10-15; `nvidia-smi -pl 200`, does not survive a host reboot)
```

Adopted 2026-10-14 after it passed the Phase 8 quality gate (H32: 19/19 at 100k, vs 18/19
for the Q5_K_M FP32 reference). Fork levers: half2 matvec incl. Q6_K (H28, H34), F16
AllReduce payload (H31/H34), cuBLAS ALGO3 for n ≥ 512 (H33, H35), full-duplex AllReduce copies (H36, bit-identical, so H32 still holds), 4-lane Q6_K matvec (H41b; fp32 reorder only, ~1e-6 relative, PPL not worse; H32 not re-run), branch-free Q4_K/Q5_K scale unpack (H43, output identical, no flag), Q3_K + IQ4_NL on mmvh (H44, `LOWBIT`, output identical, KLD gate passed), drafter LM head trimmed to the first 98,304 vocab rows (H45; lossless by construction since the target verifies every draft, output identical, acceptance unchanged), prefill flash attention through cuBLAS GEMMs at depth (H46, `FA_GEMM`: +19.2% prefill at 98k; lossy, so the H32 haystack was re-run: 19/19). The table is at 175 W. H15 (2026-10-15): 200 W adds +4.4% prefill at 98k and +4.5% at 16k, decode flat; 220 W adds +6.4% at 16k, 250 W nothing more (HYPOTHESES H15 steps 1–4). f16 KV instead of q8_0 (H48b, 2026-10-15, 200 W): 98k decode step 88.7 vs 98.4 ms (−9.9%, REPS=3 each), decode 32.8 vs 28.4 t/s, prefill unchanged, 13,573 MiB/card; f16 is the reference KV precision, the haystack is re-checked in the final report.

Fork tidy + rebase onto mainline 7182ef7 (2026-10-16/17, H56, H56b, H57): adopted levers are default-on for sm_60 (older fork builds need the `=1` env vars; tag `pre-tidy-2026-10-16`). The 100k haystack is still 19/19. Per-step decode is +0.7% and 16k prefill +1.4% (397 t/s). The greedy text changed upstream, so 16k acceptance on the bench prompt moved 75.9% → 73.5% (16k decode 46.8 t/s on that prompt). The table below is the pre-rebase REPS=10 final report.

The trimmed drafter is generated, not downloaded: `python3 /root/llama.cpp-Pascal/scripts/pascal-trim-mtp-head.py /root/mtp-Qwen3.8-27B-Q4_0.gguf /root/mtp-Qwen3.8-27B-Q4_0-head96k.gguf 98304` (needs fork cfea757+fd90268; stock builds reject the shorter head).

| Depth | Decode | Prefill | Acceptance | VRAM/card | Peak | Run |
|---|---|---|---|---|---|---|
| **Final report** (200 W, f16 KV) | **16k 47.54 ±0.09 · 100k 33.07 ±0.07 t/s** | **16k 392.1 · 100k 288.1 t/s** | — | 13,573 MiB (100k) | 70 °C | `r10s-fork-16k` (REPS=10), `r10s-fork-100k` (9 reps); stock `r10s-stock-16k` 28.71 / 318.3, `r10s-stock-100k` (q8_0, 3 reps) 20.87 / 214.3. Full table: RUNLOG 2026-10-15 → 16 |
| 98k (200 W, f16 KV) | 32.83 t/s | 288.1 t/s (~340 s TTFT) | 64.7% | 13,573 MiB | 70 °C | `h48-f16kv-98k` + `-r23` (REPS=3; q8_0 control `p15-200w-98k` + `-r3` 28.42 / 287.9 / 60.5%, 12,331 MiB) |
| 98k | 28.38 t/s | 275.75 t/s (~355 s TTFT) | 60.5% | 12,331 MiB | 67 °C | `h46-fagemm-100k` (q8_0 KV, REPS=3; control `h46-ctrl-100k` 29.65 / 231.36 / 65.3%; decode dip is the changed greedy text drafting worse, not the kernel, which only runs for n_q ≥ 512) |
| 16k | 47.47 t/s | 376.4 t/s | 75.9% | 11,157 MiB | 65 °C | `p15-175w-16k` (f16 KV, REPS=2, incl. H46; pre-H46 `h45-head96k-16k` 46.35 / 365.8 / 72.8%, REPS=3) |

Every lever behind that line, and every number, is in
[RESULTS.md](RESULTS.md). What is still open is in RUNBOOK §7.

## Status

| Phase | State |
|---|---|
| 0 smoke · 1 engine baseline · 2 drafters · 6 transport/placement | Done |
| 3 quant sweep | Folded into Phase 8 |
| 4 hypothesis tests | Ongoing — status table in [HYPOTHESES.md](HYPOTHESES.md) |
| 5 agentic web build | Tooling ready, `fork` engine added. First run aborted (pi read the repo CLAUDE.md and pushed a commit); fixed, re-run `scripts/run-p5-fork.sh` pending ([WEB_BENCH.md](WEB_BENCH.md)) |
| 7 prefill / TTFT / 100k | Done: H13, H14, H24, H26, H19 (cache reuse confirmed at defaults), H46 (GEMM FA), H15 (power cap: 200 W chosen) |
| 8 quality gate | Tier 1 (H30 KLD) and tier 2 (H32 100k haystack) passed ([QUALITY-PLAN.md](QUALITY-PLAN.md)) |

## Documents

| File | What's in it |
|---|---|
| [RUNBOOK.md](RUNBOOK.md) | **Start here.** Procedure, safety, failure modes, git protocol, **work queue** (§7: open / parked / closed) |
| [RESULTS.md](RESULTS.md) | **All the data.** Table A (`llama-bench`), Table B (`llama-server`), Table C (lever verdicts) |
| [METHODOLOGY.md](METHODOLOGY.md) | Hardware, engines, models, fixed parameters, flag syntax per engine |
| [HYPOTHESES.md](HYPOTHESES.md) | H1–H47: claim, reasoning, test, result. The narrative behind every verdict |
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
