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

## Phase 4 — hypothesis tests

| Hypothesis | Evidence | Verdict |
|---|---|---|
| H2 (XL split-stall) | _pending_ | |
| H3 (stock quant degradation) | _pending_ | |
| H4 (TurboQuant KV penalty) | _pending_ | |
| H5 (NCCL harmful) | `phase0-ik-graph-q6k` | **Partly confirmed** — fails outright, not merely slower. See HYPOTHESES.md |
| H7 (dispatch bug) | _pending_ | |
| H8 (DFlash2 usable) | `h8-loadcheck-df2q4` — runs on both GPUs, 57.0% acceptance, clean output | **Works.** Speedup not yet quantified — needs mainline's own no-drafter control |
| H9 (Q8 vs Q4 drafter) | _pending_ | No longer gated |
