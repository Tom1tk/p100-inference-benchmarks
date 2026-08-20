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
| _pending_ | | | | | | | | | | |

Cells needed: {pflash, buun, ik} × {none, layer}, plus ik × graph once
unblocked.

---

## Phase 2 — MTP on/off

| Engine | Quant | MTP | tg128 | Speedup | Acceptance rate | Notes |
|---|---|---|---|---|---|---|
| _pending_ | | | | | | |

---

## Phase 3 — quant sweep

| Engine | Split | Quant | pp16384 | tg128 | VRAM | Notes |
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
