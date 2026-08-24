# Results

Curated results. The machine-readable source of truth is
`results/all-results.csv`; these tables are for reading.

All figures t/s. Fixed parameters per [METHODOLOGY.md](METHODOLOGY.md) §6
unless a row says otherwise.

---

## Phase 0 — smoke tests

Single-round (`-r 1`) validation runs. **Not comparable to Phase 1 data** —
one repetition, no stddev. Purpose was to validate cooling before committing
to the full matrix.

| Run | Engine | Quant | Split | pp2048 | pp4096 | pp8192 | pp16384 | tg128 | Peak temp | Outcome |
|---|---|---|---|---|---|---|---|---|---|---|
| `phase0-pflash-layer-q6k` | pflash | UD-Q6_K_M | layer | 187.99 | 197.04 | 198.06 | 193.46 | 9.46 | 63°C | ✅ Clean |
| `phase0-ik-graph-q6k` | ik | UD-Q6_K_M | graph | — | — | — | — | — | 50°C | ❌ NCCL abort |

Notes:

- **pflash/layer:** both GPUs alternated to 100% utilization, confirming the
  layer split engages both cards. Prefill is essentially flat from 4k to 16k
  (197 → 193 t/s), which is a healthier depth curve than expected. `pp0`
  produced no row — expected, `-p 0` is a no-op.
- **ik/graph:** aborted with `ncclAllReduce failed with status 1` at
  `reduce.cu:169` on the first prefill test, before any timing. GPUs never
  left idle. See H5 — this is the current blocker.

**Cooling verdict: adequate.** 63°C peak under a full prefill sweep against an
83°C throttle leaves 20°C of headroom. Full matrix runs are safe.

---

## Phase 1 — engine baseline

`-r 3`, no MTP, no TurboQuant. The apples-to-apples comparison.

| Engine | Quant | Split | pp2048 | pp4096 | pp8192 | pp16384 | tg128 | VRAM (GPU0/GPU1) | Peak temp | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| pflash | UD-Q6_K_M | layer | 188.46 | 197.11 | 198.21 | 193.38 | 9.45 | — | 65°C | `-r 3`. `none` (single-GPU reference) blocked — 21.5GB model doesn't fit one 16GB card, `failed to load model`. Superseded: use `UD-IQ3_S` for the single-GPU leg |

Cells needed: {pflash, buun, ik} × {none, layer}, plus ik × graph once
unblocked.

---

## Phase 2 — drafters on/off

| Engine | Quant | Drafter | tg128 | Speedup | Acceptance rate | Notes |
|---|---|---|---|---|---|---|
| _pending_ | | | | | | |

### Preliminary — not part of the matrix

One-prompt load check, **not** `llama-bench`, **not** comparable to the rows
above or to Phase 1. Recorded because it is the first evidence DFlash2 runs here.

| Engine | Quant | Drafter | Split | Decode | Acceptance | Notes |
|---|---|---|---|---|---|---|
| mainline | UD-Q6_K_M | DFlash2-Q4_K_M | layer | 16.10 t/s | 239/419 = 57.0% | `-c 4096`, 300 tok, temp 0, one prompt. Load 4m40s, 13.8/15.0 GiB, 49°C |

---

## Phase 3 — quant sweep

| Engine | Split | Quant | pp16384 | tg128 | VRAM | Notes |
|---|---|---|---|---|---|---|
| _pending_ | | `IQ3_S` | | | | Run on both `layer` and `none` — the only single-GPU-capable target |
| _pending_ | | `Q4_K_M` | | | | |
| _pending_ | | `Q5_K_M` | | | | |
| _pending_ | | `Q6_K_M` | | | | |

---

## Phase 5 — agentic web-build benchmark

Real one-shot agentic work, not synthetic prompts. Procedure and quality
criteria in [WEB_BENCH.md](WEB_BENCH.md). Machine-readable: `results/web-bench.csv`.

### Throughput and cost

| Label | Idx | Engine | Model | Drafter | Total time | Tokens gen | Prefill avg/min/max | Decode avg/min/max | Peak temp |
|---|---|---|---|---|---|---|---|---|---|
| _pending_ | | | | | | | | | |

Drafter arms per config: none (control) · MTP · DFlash2-Q4 · DFlash2-Q8.
The DFlash2 arms require `engine = mainline`, which needs its own no-drafter
control row so the drafter comparison isn't confounded with the engine. See H8.

### Quality

Hand-scored after each run. A model can be fast and still fail here.

| Label | Stage 1 | Stage 2 | Stage 3 | One-shot? | Tool calls | Verdict |
|---|---|---|---|---|---|---|
| _pending_ | | | | | | |

---

## Phase 6 — H11: drafter placement

Target `UD-Q4_K_M`, `-sm layer`, `-c 4096`, `-fa 1`, `-ngl 99`, 400 tokens,
temp 0, seed 42, 3 reps. Only the drafter and its `-devd` placement vary.
Raw: `results/h11-placement.csv`, `results/h11/*.jsonl`.

`decode` is the mean of reps 2-3; rep 1 is discarded because a cold page cache
depresses it (see the mtp-default 13.91 outlier). **Ignore the `prefill_tps`
column** — `cache_prompt: false` did not defeat the server's slot-level LCP
prefix reuse (`f_sim_best = 1.000` in the logs), so reps 2-3 skip most of
prefill. Decode and acceptance are unaffected, and they are what H11 turns on.

| Arm | Placement | Decode t/s | vs control | Acceptance | VRAM 0/1 (MiB) | Peak |
|---|---|---|---|---|---|---|
| no drafter | — | **12.79** | — | — | 7693 / 8769 | 59°C |
| MTP-Q4_0 | default | 14.11 | +10.3% | 44.9% | 8647 / 10997 | 59°C |
| MTP-Q4_0 | `-devd CUDA0` | 14.12 | +10.4% | 44.9% | 9569 / 9983 | 58°C |
| MTP-Q4_0 | `-devd CUDA1` | 14.11 | +10.3% | 44.9% | 8647 / 10905 | 60°C |
| DFlash2-Q4_K_M | default | 14.46 | +13.0% | 45.8% | 9679 / 10619 | 59°C |
| DFlash2-Q4_K_M | `-devd CUDA0` | — | — | — | — | ❌ **abort** |
| DFlash2-Q4_K_M | `-devd CUDA1` | **14.48** | **+13.2%** | 45.8% | 8647 / 11361 | 60°C |
| DFlash2-Q8_0 | default | 14.44 | +12.9% | 44.4% | 10291 / 10879 | 59°C |
| DFlash2-Q8_0 | `-devd CUDA0` | — | — | — | — | ❌ **abort** |
| DFlash2-Q8_0 | `-devd CUDA1` | 14.46 | +13.1% | 44.4% | 8647 / 12233 | 60°C |

**Placement does not affect decode throughput.** Within each drafter the spread
is 0.01-0.02 t/s (≤0.15%), far inside run-to-run noise, while VRAM demonstrably
moved between cards (GPU0 ranges 8647-10291 MiB). The flag works; decode does
not care. **H11 is refuted.**

The control variable behaved: acceptance is identical across placements of the
same drafter, to the exact draft-token count (MTP 228/513 in all three arms).
That is what makes this a real null rather than a noisy one — had placement
perturbed anything but locality, acceptance would have moved too.

### Depth check — does the null survive a real context?

The table above decodes at ~450 tokens of actual depth: `-c 4096` only sizes the
KV *allocation*, and a short prompt plus 400 generated tokens never fills it.
Since placement is a locality question and inter-GPU traffic scales with depth,
the null was re-tested at `-c 16384` driven by a **14.5k-token prompt** (the
repo's own docs — real varied text, deterministic), DFlash2-Q4, 3 reps.

| Placement | Decode t/s (mean 3) | Prefill t/s | Acceptance | VRAM 0/1 (MiB) | GPU1 free | Peak |
|---|---|---|---|---|---|---|
| `default` | **17.10** | 160.0 | 64.8% (263/406) | 10201 / 11087 | 5182 | 63°C |
| `-devd CUDA1` | 17.06 | 160.1 | 64.8% (263/406) | 9079 / 11887 | **4382** | 63°C |

**The null holds at depth** — 0.2% apart, with `default` fractionally ahead, and
acceptance identical to the token (263/406 in both). H11 is refuted at both
~450 and ~14.5k tokens of context.

Note the VRAM column: at this depth `-devd CUDA1` gives up **800 MiB of headroom
on the card that is already fullest** (2808 MiB imbalance vs 886 for `default`)
in exchange for nothing measurable. That is the argument against pinning, and it
grows with context.

Two incidental observations, neither part of H11:

- **Decode is *faster* at 14.5k depth than at 450** (17.10 vs 14.46 t/s),
  because acceptance rose from 45.8% to 64.8%. Summarising a supplied document
  is far more predictable than free-form generation, and the drafter wins more
  often. Depth cost less than predictability gained — worth remembering when
  reading any single-prompt acceptance figure, including this one.
- **Prefill is measurable here**: ~160 t/s over 14.5k tokens, stable across all
  six runs. Unlike the short-prompt rows, these numbers look like genuine
  reprocessing rather than prefix reuse.

### Two findings that were not what H11 was looking for

**1. `-devd CUDA0` aborts with any DFlash2 drafter.** Reproducible on both
quants, at load, before any token:

```
llama_init_from_model: failed to initialize the context: dflash requires ctx_other to be set
srv load_model: [spec] failed to measure draft model memory
ggml-backend.cpp:930: pre-allocated tensor (output.weight) in a buffer (CUDA1)
                      that cannot run the operation (NONE)   -> ggml_abort
```

Cause: the DFlash2 drafter GGUF ships **neither `output.weight` nor
`token_embd.weight`** — it borrows the target's through `cparams.ctx_other`.
Under `-sm layer` the target's `output.weight` lives on the last device, CUDA1.
`-devd CUDA0` restricts the draft context to CUDA0, so the borrowed tensor sits
in a buffer the draft scheduler may not use, and ggml aborts.

This explains every cell: MTP carries its own `output.weight` and
`token_embd.weight`, so all three of its placements load; DFlash2 survives
`default` (both devices permitted) and `CUDA1` (where the tensor already is),
and only ever fails on `CUDA0`. It is an upstream bug in PR #27342, not a P100
or VRAM limit — GPU0 had ~6 GiB free at the time.

**2. The Q8 drafter is not better than the Q4 drafter** (H9 evidence).
DFlash2-Q8_0 accepted 228/513 = 44.4%; DFlash2-Q4_K_M accepted 231/504 = 45.8%,
and was fractionally faster (14.48 vs 14.46 t/s) while using **0.87 GiB less
VRAM**. One prompt at temp 0, so this is exact but not yet general — it needs
more prompts before H9 is called. It does mean the Q8 drafter has no measured
advantage to justify its footprint.

### First honest speculative-decoding speedups

With mainline's own no-drafter control at 12.79 t/s, the real numbers are
**+10.3% (MTP)** and **+13.0% (DFlash2)** — not the ~70% the earlier
uncontrolled 9.45 → 16.10 t/s comparison implied. That older pair differed in
engine, tool, and context depth; this one differs only in the drafter.

Note the gap between acceptance and speedup: DFlash2 accepts 45.8% of drafted
tokens and returns 13%. On P100 the verify pass is not cheap enough for
acceptance to translate proportionally.

---

## Phase 4 — hypothesis tests

| Hypothesis | Evidence | Verdict |
|---|---|---|
| H2 (XL split-stall) | _pending_ | |
| H3 (stock quant degradation) | _pending_ | |
| H4 (TurboQuant KV penalty) | _pending_ | |
| H5 (NCCL harmful) | `phase0-ik-graph-q6k` | **Partly confirmed** — fails outright, not merely slower. See HYPOTHESES.md |
| H7 (dispatch bug) | _pending_ | |
| H8 (DFlash2 usable) | `h8-loadcheck-df2q4` — runs on both GPUs, 57.0% acceptance, clean output | **Works.** Speedup not yet quantified — needs mainline's own no-drafter control |
| H9 (Q8 vs Q4 drafter) | `h11-df2q8-*` vs `h11-df2q4-*` | **Leaning no** — Q8 accepted 44.4% vs Q4's 45.8%, no faster, +0.87 GiB. One prompt; needs more |
| H11 (drafter placement) | `results/h11-placement.csv` | **Refuted** — ≤0.15% spread across placements. Found a `-devd CUDA0` abort with DFlash2 instead |
