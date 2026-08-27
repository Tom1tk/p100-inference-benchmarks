# Results

Curated results. The machine-readable source of truth is
`results/all-results.csv`; these tables are for reading.

All figures t/s. Fixed parameters per [METHODOLOGY.md](METHODOLOGY.md) §6
unless a row says otherwise.

**Note:** `results/all-results.csv` is **stale** — it stops at the Phase 1 and
H10 runs. The complete raw data is the per-run files in `results/raw/` plus
`results/h11-placement.csv`; the master matrix below is generated from those.

---

# Master results matrix

**Everything measured on this rig, in one place.** Rebuilt 2026-08-27 directly
from `results/raw/*.csv` and `results/h11-placement.csv`, so it is the tables
below that are derived, not this one. Two harnesses measure different things and
their numbers are **not interchangeable**:

- **Table A — `llama-bench`**: synthetic, no drafter, no server. Good for
  engine/split/quant/ubatch comparisons. `pp*` = prefill at that prompt length,
  `tg128` = decode.
- **Table B — `llama-server`**: real HTTP requests, 400 tokens, temp 0, seed 42.
  The only harness that measures drafters, acceptance and real VRAM. **This is
  the one the serve command is built from.**

Hardware is constant throughout: 2x Tesla P100-PCIE-16GB (16,269 MiB each,
732.2 GB/s, sm_60, no NVLink, PCIe 3.0 PHB), Xeon E5-2680 v4, **175 W cap**,
`-ngl 99 -fa 1 -t 8`. Any row marked GPUs=1 used `-dev CUDA0`.

## Table A — `llama-bench` (no drafter)

_Generated: `python3 scripts/build-matrix.py`. Regenerate and paste after any new sweep._

| Run | Engine | Quant | `-sm` | GPUs | `-b` | `-ub` | KV | pp 2048 | pp 4096 | pp 8192 | pp 16384 | pp 65536 | pp 100000 | tg128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `h10-p2p-off-q4km` | rebased | Q4_K_M | tensor | 2 | 2048 | 512 | f16 | **222.8** | **223.4** | **214.9** | **208.3** | — | — | **20.3** |
| `h10-p2p-on-q4km` | rebased | Q4_K_M | tensor | 2 | 2048 | 512 | f16 | **221.9** | **222.8** | **214.2** | **207.6** | — | — | **20.9** |
| `h13-prefill-depth-q4km` | rebased | Q4_K_M | tensor | 2 | 2048 | 2048 | f16 | — | — | — | **327.1** | **250.1** | **215.4** | — |
| `h14-ubatch-sweep-hi-q4km` | rebased | Q4_K_M | tensor | 2 | 8192 | 1024 | f16 | — | — | **271.9** | — | — | — | — |
| `h14-ubatch-sweep-hi-q4km` | rebased | Q4_K_M | tensor | 2 | 8192 | 2048 | f16 | — | — | **342.9** | — | — | — | — |
| `h14-ubatch-sweep-hi-q4km` | rebased | Q4_K_M | tensor | 2 | 8192 | 4096 | f16 | — | — | **347.5** | — | — | — | — |
| `h14-ubatch-sweep-q4km` | rebased | Q4_K_M | tensor | 2 | 2048 | 128 | f16 | **197.0** | **194.8** | — | — | — | — | — |
| `h14-ubatch-sweep-q4km` | rebased | Q4_K_M | tensor | 2 | 2048 | 256 | f16 | **249.7** | **241.7** | — | — | — | — | — |
| `h14-ubatch-sweep-q4km` | rebased | Q4_K_M | tensor | 2 | 2048 | 512 | f16 | **218.9** | **221.8** | — | — | — | — | — |
| `h14-ubatch-sweep-q4km` | rebased | Q4_K_M | tensor | 2 | 2048 | 1024 | f16 | **278.8** | **276.5** | — | — | — | — | — |
| `h14-ubatch-sweep-q4km` | rebased | Q4_K_M | tensor | 2 | 2048 | 2048 | f16 | **357.5** | **351.1** | — | — | — | — | — |
| `h17-control-nofix` | rebased | Q4_K_M | tensor | 2 | 2048 | 512 | f16 | **222.7** | — | — | **208.4** | — | — | **20.4** |
| `h17-patched-fp32` | rebased+h17 | Q4_K_M | tensor | 2 | 2048 | 512 | f16 | **193.0** | — | — | **182.8** | — | — | **20.4** |
| `h25-iq3-1card-kv-q4` | rebased | IQ3_S | none | 1 | 2048 | 2048 | q4_0 | — | **202.1** | — | **184.0** | — | — | **11.2** |
| `h25-iq3-1card-kv-q8` | rebased | IQ3_S | none | 1 | 2048 | 2048 | q8_0 | — | **202.4** | — | **184.1** | — | — | **11.2** |
| `h25-iq3-1card` | rebased | IQ3_S | none | 1 | 2048 | 512 | f16 | — | **132.3** | — | **125.9** | — | — | **11.4** |
| `h25-iq3-1card` | rebased | IQ3_S | none | 1 | 2048 | 2048 | f16 | — | **201.0** | — | **183.8** | — | — | **11.4** |
| `h25-iq3-2card-tensor` | rebased | IQ3_S | tensor | 1 | 2048 | 512 | f16 | — | **131.5** | — | **125.4** | — | — | **11.4** |
| `h25-iq3-2card-tensor` | rebased | IQ3_S | tensor | 1 | 2048 | 2048 | f16 | — | **200.1** | — | — | — | — | — |
| `phase1-buun-layer-q4km` | buun | Q4_K_M | layer | 2 | 2048 | 512 | f16 | **178.5** | **194.3** | **200.7** | **198.7** | — | — | **13.1** |
| `phase1-buun-tensor-q4km` | buun | Q4_K_M | tensor | 2 | 2048 | 512 | f16 | **193.5** | **191.9** | **188.6** | **182.7** | — | — | **20.7** |
| `phase1-iknonccl-graph-q4km` | ik | Q4_K_M | graph | 2 | 2048 | 512 | f16 | **199.1** | **198.4** | **179.6** | **153.0** | — | — | **22.1** |
| `phase1-iknonccl-layer-q4km` | ik | Q4_K_M | layer | 2 | 2048 | 512 | f16 | **116.5** | **109.5** | **97.0** | **80.1** | — | — | **13.5** |
| `phase1-mainline-layer-q4km` | mainline | Q4_K_M | layer | 2 | 2048 | 512 | f16 | **180.4** | **191.0** | **192.1** | **188.7** | — | — | **12.9** |
| `phase1-mainline-tensor-internal-q4km` | mainline | Q4_K_M | tensor | 2 | 2048 | 512 | f16 | **214.8** | **219.1** | **212.7** | **206.3** | — | — | **20.3** |
| `phase1-mainline-tensor-none-q4km` | mainline | Q4_K_M | tensor | 2 | 2048 | 512 | f16 | **214.7** | **219.1** | **212.7** | **206.4** | — | — | **20.4** |
| `phase1-pflash-layer-q4km` | PFlash* | Q4_K_M | layer | 2 | 2048 | 512 | f16 | **184.3** | **191.6** | **192.3** | **187.9** | — | — | **12.5** |
| `phase1-pflash-layer-q6k` | PFlash* | Q6_K_M | layer | 2 | 2048 | 512 | f16 | **188.5** | **197.1** | **198.2** | **193.4** | — | — | **9.5** |
| `phase1-rebased-layer-q4km` | rebased | Q4_K_M | layer | 2 | 2048 | 512 | f16 | **180.8** | **190.8** | **192.1** | **187.1** | — | — | **12.9** |
| `phase1-rebased-tensor-q4km` | rebased | Q4_K_M | tensor | 2 | 2048 | 512 | f16 | **222.6** | **223.6** | **215.0** | **208.4** | — | — | **20.3** |

`*` PFlash is closed (sm_60-incompatible, no v2) — rows kept for the record only.

**Reading notes.**

- **`-ub` is the single biggest prefill lever, and 512 is a local minimum.**
  Look down the `h14-ubatch-sweep` block: 128 -> 197, 256 -> 250, **512 -> 219**,
  1024 -> 279, 2048 -> **357**. The default is worse than its neighbours in *both*
  directions. Every pre-H14 number in this repo was taken at 512 and is ~35% low.
- **`-ub 4096` is not worth the `-b 8192` it needs**: 347.5 vs 342.9 at pp8192,
  and `-ub 8192` aborts (the crash backtrace is still in
  `h14-ubatch-sweep-hi-q4km.csv`).
- **`-sm tensor` beats `-sm layer` on both axes** on the rebased engine: 208 vs
  187 prefill, 20.3 vs 12.9 decode. That decode gap (**+58%**) is the largest
  single-flag effect in the table.
- **`ik` wins decode and loses everything else** (22.1 tg128, best in table, but
  153 pp16384 and no tensor split at all). It also collapses hardest with depth.
- **P2P makes no difference**: `h10-p2p-on` vs `off` agree within 0.6% on every
  cell. Settled, do not re-run.
- **KV quant is nearly free** (`h25-iq3-1card-kv-*`): f16/q8_0/q4_0 give
  183.8/184.1/184.0 prefill and 11.38/11.21/11.18 decode. `q4_0` buys nothing
  over `q8_0`.
- `h25-iq3-2card-tensor` is mis-named: it ran `-sm tensor` with `-dev CUDA0`,
  i.e. one card. It matches the `-sm none` single-card row to within 0.4%, which
  is the useful result — **`-sm` is a no-op when only one device is visible.**

## Table B — `llama-server` (real requests, drafters, VRAM)

### H17 — the sm_60 FP16 fast-path A/B (2026-08-27)

Same tree, same flags, one session; the only difference is the build. This is
the cleanest paired comparison in the repo and it reprices the buun row above.

| | control (fix off) | patched (fix on) | delta | buun (fix on) |
|---|---|---|---|---|
| pp2048 | 222.72 | **193.01** | **-13.3%** | 193.50 |
| pp16384 | 208.44 | **182.76** | **-12.3%** | 182.75 |
| tg128 | 20.357 | **20.430** | **+0.36%** | 20.67 |

Std devs 0.01-0.14 t/s. Patched rebased lands within **0.3%** of buun on prefill
at both depths — so buun's prefill "deficit" was this fix all along. Its decode
lead survives at ~1.2% and is a genuine engine difference.

**Side effect:** halving the GEMM rate cost only 12-13% of prefill, implying
**~15% of prefill time is GEMM-rate-bound**. That is the first number anyone has
put on H18, and it came free.

| Run | Engine | Quant | `-sm` | GPUs | `-ub` | KV | Drafter | Ctx | Decode | Prefill | Accept | VRAM/card | Peak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `smoke-tensor-none` | rebased | Q4_K_M | tensor | 2 | 512 | f16 | none | 4k | 20.13 | 66.5 | — | 8183/8183 | 50 C |
| `smoke-tensor-mtp` | rebased | Q4_K_M | tensor | 2 | 512 | f16 | MTP-Q4_0 | 4k | 32.60 | 55.5 | 90.0% | 9827/9827 | 53 C |
| `smoke-tensor-df2q4` | rebased | Q4_K_M | tensor | 2 | 512 | f16 | DFlash2-Q4 | 4k | **FAILED(load)** | — | — | — | — |
| `p2-tensor-none` | rebased | Q4_K_M | tensor | 2 | 512 | f16 | none | 4k | 20.30 | 67.6 | — | 8183/8183 | 65 C |
| `p2-tensor-mtp` | rebased | Q4_K_M | tensor | 2 | 512 | f16 | MTP-Q4_0 | 4k | 21.27 | 58.4 | 40.1% | 9827/9827 | 66 C |
| `p2-tensor-none-16k` | rebased | Q4_K_M | tensor | 2 | 512 | f16 | none | 16k | 19.75 | 199.0 | — | 8579/8579 | 68 C |
| `p2-tensor-mtp-16k` | rebased | Q4_K_M | tensor | 2 | 512 | f16 | MTP-Q4_0 | 16k | 29.26 | 198.4 | 73.3% | 10235/10235 | 70 C |
| `p2-layer-none-16k` | rebased | Q4_K_M | layer | 2 | 512 | f16 | none | 16k | 12.35 | 183.9 | — | 8125/9201 | 66 C |
| `p2-layer-mtp-16k` | rebased | Q4_K_M | layer | 2 | 512 | f16 | MTP-Q4_0 | 16k | 19.05 | 162.9 | 77.5% | 9079/11525 | 65 C |
| `h11-none-control` | rebased | Q4_K_M | layer | 2 | 512 | f16 | none | 4k | 12.79 | _n/a_ | — | 7693/8769 | 59 C |
| `h11-mtp-default` | rebased | Q4_K_M | layer | 2 | 512 | f16 | MTP-Q4_0 | 4k | 14.04 | _n/a_ | 44.6% | 8647/10997 | 59 C |
| `h11-mtp-cuda0` | rebased | Q4_K_M | layer | 2 | 512 | f16 | MTP, `-devd CUDA0` | 4k | 14.16 | _n/a_ | 44.6% | 9569/9983 | 58 C |
| `h11-mtp-cuda1` | rebased | Q4_K_M | layer | 2 | 512 | f16 | MTP, `-devd CUDA1` | 4k | 14.15 | _n/a_ | 44.6% | 8647/10905 | 60 C |
| `h11-df2q4-default` | rebased | Q4_K_M | layer | 2 | 512 | f16 | DFlash2-Q4 | 4k | **14.44** | _n/a_ | 45.8% | 9679/10619 | 59 C |
| `h11-df2q4-cuda1` | rebased | Q4_K_M | layer | 2 | 512 | f16 | DFlash2-Q4, CUDA1 | 4k | 14.48 | _n/a_ | 45.8% | 8647/11361 | 60 C |
| `h11-df2q4-cuda0` | rebased | Q4_K_M | layer | 2 | 512 | f16 | DFlash2-Q4, CUDA0 | 4k | **FAILED(load)** | — | — | — | — |
| `h11-df2q8-default` | rebased | Q4_K_M | layer | 2 | 512 | f16 | DFlash2-Q8 | 4k | 14.44 | _n/a_ | 44.4% | 10291/10879 | 59 C |
| `h11-df2q8-cuda1` | rebased | Q4_K_M | layer | 2 | 512 | f16 | DFlash2-Q8, CUDA1 | 4k | 14.46 | _n/a_ | 44.4% | 8647/12233 | 60 C |
| `h11-df2q8-cuda0` | rebased | Q4_K_M | layer | 2 | 512 | f16 | DFlash2-Q8, CUDA0 | 4k | **FAILED(load)** | — | — | — | — |
| `h11-depth16k-df2q4-default` | rebased | Q4_K_M | layer | 2 | 512 | f16 | DFlash2-Q4 | 16k | 17.10 | _n/a_ | 64.8% | 10201/11087 | 63 C |
| `h11-depth16k-df2q4-cuda1` | rebased | Q4_K_M | layer | 2 | 512 | f16 | DFlash2-Q4, CUDA1 | 16k | 17.06 | _n/a_ | 64.8% | 9079/11887 | 63 C |
| `p3-buun-mtp-16k-f16` | buun | Q4_K_M | tensor | 2 | 512 | f16 | MTP-Q4_0 | 16k | **28.12** | 178.2 | 81.6% | 10031/11521 | 70 C |
| `p3-buun-mtp-16k-h7` | buun | Q4_K_M | tensor | 2 | 512 | turbo3/f16 | MTP-Q4_0 | 16k | 24.76 | 177.7 | 74.2% | 9835/11325 | 70 C |
| `p3-buun-mtp-16k-t3t3` | buun | Q4_K_M | tensor | 2 | 512 | turbo3 | MTP-Q4_0 | 16k | 24.09 | 177.8 | 77.2% | 9637/11127 | 70 C |
| `h24-ub512-mtp-16k` | rebased | Q4_K_M | tensor | 2 | 512 | f16 | MTP-Q4_0 | 16k | **29.39** | 198.5 | 73.3% | 10235/10235 | 67 C |
| `h24-ub2048-mtp-16k` | rebased | Q4_K_M | tensor | 2 | **2048** | f16 | MTP-Q4_0 | 16k | 27.41 | **310.5** | 66.4% | 10973/10973 | 68 C |
| `h25-iq3-1card-none-16k` | rebased | IQ3_S | none | **1** | 2048 | q8_0 | none | 16k | 10.42 | 183.8 | — | 12601 | 69 C |
| `h25-iq3-1card-mtp-16k` | rebased | IQ3_S | none | **1** | 2048 | q8_0 | MTP-Q4_0 | 16k | **OOM** | — | — | 16029 at load | — |
| `h26-100k-mtp-kvq8` | rebased | Q4_K_M | tensor | 2 | 2048 | **q8_0** | MTP-Q4_0 | **100k** | **20.02** | **207.4** | 60.3% | 12147/12147 | 70 C |

**Reading notes.**

- **`_n/a_` prefill in the `h11-*` block is not missing data — it is invalid
  data.** `cache_prompt: false` did not defeat the server's slot-level LCP prefix
  reuse (`f_sim_best = 1.000` in the logs), so reps 2-3 skipped most of prefill.
  Decode and acceptance are unaffected. Never quote an h11 prefill number.
- **`smoke-tensor-mtp` (32.60 t/s, 90% acceptance) is a single rep on a cold
  cache and should not be treated as a record.** `p2-tensor-mtp` re-ran the same
  config over 3 reps and got 21.27 at 40.1% acceptance.
- **Acceptance is strongly depth-dependent**, and it is the mechanism behind most
  of the decode spread here: MTP goes 40.1% at 4k -> 73.3% at 16k, and DFlash2-Q4
  goes 45.8% -> 64.8%. Any drafter conclusion drawn at 4k does not transfer.
- **DFlash2-Q4 beat MTP at 4k** (14.44 vs 14.04) but has **never been measured at
  the current config** (tensor split, `-ub 2048`, 16k). That comparison is open.
- **DFlash2 cannot be pinned to CUDA0** — `-devd CUDA0` aborts at load on both
  DFlash2 quants while MTP is fine. Bug, not a speed result.
- **Q8 drafters are not worth it** (H9): DFlash2-Q8 accepted 44.4% vs Q4's 45.8%,
  identical decode, +0.87 GiB.
- **The 175 W cap is doing the thermal work.** The hottest number anywhere in this
  table is 70 C against an 83 C abort limit — and `h26` shows that holds under
  *sustained* load too: two consecutive ~8-minute 100k prefills peaked at the
  same 70 C as the 3-minute 16k runs.
- **`h26-100k-mtp-kvq8` is the deliverable config**, and the only row here at the
  target depth. Ignore the two `FAILED(request)` rows for the same label in
  `results/h11-placement.csv`: they are a harness bug (a ~340 KB prompt passed
  through argv, overflowing E2BIG so curl posted an empty body), not a model or
  VRAM failure. Fixed 2026-08-27.
- **Acceptance is not monotonic in depth.** MTP goes 40.1% (4k) -> 66.4% (16k) ->
  60.3% (100k). It peaks somewhere in the middle, so neither the 4k nor the 100k
  figure can be extrapolated from the other.

## Table C — lever verdicts at a glance

| Lever | Values tested | Verdict | Evidence |
|---|---|---|---|
| Engine | mainline, rebased, buun, `ik`, PFlash | **rebased** (mainline + DFlash2, same speed as mainline) | Table A phase1 block |
| `-sm` | `layer`, `tensor`, `graph` (`ik`), `none` (1 card) | **`tensor`** — +58% decode, +11% prefill over layer | H1/H6 |
| **sm_60 FP16 fast path** | on (stock mainline) vs off (buun / the H17 patch) | **Leave it ON** for speed — patching it out costs **12–13% of prefill** and buys only **+0.36%** decode. But it is an arithmetic *correctness* fix, so this is a speed-vs-quality trade, **not settled** | H17 |
| `GGML_CUDA_ALLREDUCE` | NCCL default, `internal`, `none` | **`none`** (default aborts under tensor split; internal == none) | H5 |
| `-ub` | 128, 256, 512, 1024, 2048, 4096, 8192 | **2048** — +63% prefill; 8192 aborts; 4096 needs `-b 8192` for +1% | H14, H24 |
| `-b` | 2048, 8192 | **2048** — only matters as a ceiling on `-ub` | H14 |
| KV `-ctk`/`-ctv` | f16, q8_0, q4_0, turbo3 | **`q8_0`** — 1.5% of decode, half the cache; q4_0 adds nothing; turbo3 costs 14.3% | H25, H4 |
| Model quant | Q4_K_M, Q6_K (dead engine), IQ3_S (1 card) | **Q4_K_M** by default, **never actually swept** | Table A |
| Drafter | none, MTP-Q4_0, DFlash2-Q4, DFlash2-Q8 | **MTP** at 16k/tensor; DFlash2 led at 4k/layer and is unretested | H8, H9, H11 |
| Drafter placement `-devd` | default, CUDA0, CUDA1 | **No effect** (<=0.15%). DFlash2+CUDA0 aborts | H11 |
| P2P | on, off | **No effect** (<=0.6%) | H10 |
| GPUs | 1, 2 | **2 required.** One card = 56% prefill, 38% real decode, cannot reach 100k | H20/H25 |
| Power cap | 175 W only | Untested above 175 W; approved to 220 W pending PSU check | H15 |
| Context depth | 2k, 4k, 8k, 16k, 64k, **100k end to end** | Serves at 100k: decode -27%, prefill -33%, acceptance -6 pts vs 16k | H13, **H26** |
| Sustained thermals | ~8 min continuous 100k prefill, x2 | **70 C peak**, same as the 3-min 16k runs. Not a constraint at 175 W | H26 |

## Table D — coverage: what has never been measured

The gaps matter more than the numbers above. Ranked by whether the answer could
change the serve command.

| Gap | State | Why it matters | Cost |
|---|---|---|---|
| ~~**100k on `llama-server`**~~ | **CLOSED 2026-08-27 (H26).** Serves at 20.02 t/s decode / 207.4 prefill / 60.3% acceptance / 12,147 MiB per card / 70 C | — | — |
| **Quality — all of it** _(now the top gap)_ | **Zero data.** No NIAH at any depth, no A/B against a baseline, Phase 5 quality table entirely `_pending_` | It is one of the four objectives and the only one with no measurement whatsoever. Every speed result above is unvalidated for output quality | NIAH tiers, cheap per run |
| ~~**Decode at depth >16k**~~ | **CLOSED 2026-08-27 (H26).** 20.02 t/s at 100k, 73% of the 16k figure | — | — |
| ~~**`q8_0` KV on two cards**~~ | **CLOSED 2026-08-27 (H26).** Works; the 100k cache cost ~1,174 MiB/card instead of ~2,350 at f16 | — | — |
| **Model quant sweep** | **Never run.** Q4_K_M throughout; the only Q6_K row is on the dead PFlash engine at `-ub 512`/layer; IQ3_S only on one card | Q5_K_M/Q6_K are the obvious quality lever, and quality is the untested objective. Also decides whether 100k fits | Phase 3, 1 run per quant |
| **Drafter re-rank at current config** | All drafter comparisons ran at `-ub 512`, 4k, `-sm layer`. DFlash2 **led** there (14.44 vs 14.04) | H24 showed acceptance is ubatch-sensitive, and H26 showed it is not monotonic in depth (40.1 -> 66.4 -> 60.3%). The MTP-vs-DFlash2 ranking is unverified at `-ub 2048`/tensor, and now doubly so at 100k | 1 arm vs a banked control |
| **Prompt-cache reuse (H19)** | Never run | Highest practical value for the actual use case: an agentic harness re-sends a growing prefix every turn. Costs a flag | 1 multi-turn run |
| **sm_60 FP16 fast-path fix (H17)** | Never run | Free by hypothesis, and it un-confounds every quality comparison between engines | Rebuild + 1 cell |
| ~~**Sustained thermals**~~ | **CLOSED 2026-08-27 (H26).** 70 C peak over two consecutive ~8-min 100k prefills — no different from the short runs | — | — |
| **Power cap >175 W (H15)** | Never tested | Approved to 220 W; prefill is compute-bound so it is the one lever that could move it without a kernel | Blocked on the user's in-person PSU meter check |
| **`--jinja` / agentic phase** | Never run (Phase 5 entirely `_pending_`) | It is the actual deployment target; throughput on synthetic prompts is a proxy for it | Long runs |
| **Concurrency / multiple slots** | Never tested (`n_slots = 4` was in force and never exercised) | Only matters if the harness issues parallel requests — ask before spending anything here | Unknown |

**Closed, do not spend runs:** single-GPU serving (H20/H25), PFlash (H21),
TurboPrefill (H16), TurboQuant KV (H4), P2P (H10), drafter placement (H11),
Q8 drafters (H9), `-ub` above 2048 (H14). Parked with reasons in
[RUNBOOK.md](RUNBOOK.md): H18, H22, H23, H12.

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

`-r 3`, no MTP, no TurboQuant, `-ngl 99 -fa 1 -t 8 -ctk f16 -ctv f16`.
The apples-to-apples comparison. All cells on **`Qwen3.8-27B-UD-Q4_K_M`**.

| Engine | Commit | Split | pp2048 | pp4096 | pp8192 | pp16384 | tg128 | Run |
|---|---|---|---|---|---|---|---|---|
| ik (no NCCL) | `8337e4c` | graph | 199.14 | 198.45 | 179.56 | 153.04 | 22.05 | 676s, peak 68C |
| ik (no NCCL) | `8337e4c` | layer | 116.49 | 109.50 | 96.99 | 80.08 | 13.54 | 1065s, peak 62C |
| pflash | `e05ff58b7` | layer | 184.31 | 191.64 | 192.33 | 187.87 | 12.53 | 685s, peak 65C |
| buun | `39d97a876` | layer | 178.47 | 194.31 | 200.73 | 198.67 | 13.10 | 659s, peak 68C |
| mainline (pr-27342) | `64f765f5` | layer | 180.38 | 190.96 | 192.14 | 188.69 | 12.88 | 684s, peak 65C |
| ik (NCCL on) | `8337e4c` | graph | — | — | — | — | — | **exit 134 after 9s** — `ncclAllReduce failed with status 1`. See H5 |
| mainline (pr-27342) | `64f765f5` | **tensor**, `ALLREDUCE=internal` | **214.83** | **219.13** | **212.70** | **206.32** | 20.34 | 608s, peak 67C |
| mainline (pr-27342) | `64f765f5` | **tensor**, `ALLREDUCE=none` | 214.68 | 219.12 | 212.71 | 206.40 | 20.38 | 608s, peak 69C |
| mainline (pr-27342) | `64f765f5` | tensor, `ALLREDUCE=nccl` (Linux default) | — | — | — | — | — | **exit 134 after 5s** — `ggml_backend_cuda_comm_allreduce_nccl`. See H5/H10 |
| **mainline rebased** | `57affa09` | **tensor**, `ALLREDUCE=internal` | **222.62** | **223.59** | **214.98** | **208.38** | 20.34 | 601s, peak 69C. Best cell in the matrix |
| mainline rebased | `57affa09` | layer | 180.85 | 190.80 | 192.09 | 187.15 | 12.89 | — |
| mainline rebased | `57affa09` | tensor, `ALLREDUCE=none`, **`P2P=1`** | 221.91 | 222.77 | 214.19 | 207.63 | **20.92** | Best decode. P2P costs ~0.3% pp, buys +2.9% tg. See H10 |
| mainline rebased | `57affa09` | tensor, `ALLREDUCE=none`, P2P off | 222.75 | 223.43 | 214.88 | 208.25 | 20.34 | Reproduces the `internal` cell above to within 0.06% |
| **buun** | `39d97a876` | **tensor**, `ALLREDUCE=none` | 193.50 | 191.86 | 188.57 | 182.75 | 20.67 | 684s, peak 70C. Tensor works on buun. **The 12–14% prefill gap is NOT an engine difference** — see H17 below: it is the sm_60 FP16 fix, which buun carries and rebased does not. Patch rebased and it lands on 193.01 / 182.76, i.e. **on top of this row** |

Earlier Q6_K_M reference cell, kept for continuity (different quant — not
comparable to the rows above):

| Engine | Quant | Split | pp2048 | pp4096 | pp8192 | pp16384 | tg128 | Peak temp | Notes |
|---|---|---|---|---|---|---|---|---|---|
| pflash | UD-Q6_K_M | layer | 188.46 | 197.11 | 198.21 | 193.38 | 9.45 | 65°C | `none` (single-GPU reference) blocked — 21.5GB model doesn't fit one 16GB card, `failed to load model`. Superseded: use `UD-IQ3_S` for the single-GPU leg |

### What Phase 1 established

**1. Splitting the *computation* is what buys decode — not which fork does it.**
Both compute-splitting modes break out of the layer-split band: `ik`'s graph at
22.05 and mainline's tensor at 20.34, against 12.53–13.54 for *every* layer
implementation in *every* engine. Layer split is the thing that's slow; the
engine is incidental.

**2. `-sm tensor` on mainline is the best cell in the matrix.**
Fastest prefill at *every* depth (214.83 / 219.13 / 212.70 / 206.32), and it
holds shape: −6% from its 4k peak out to 16k. It beats `ik` graph by **+35% at
pp16384** and gives up only **7.8%** of graph's decode. Both cards run at
97–99% utilisation simultaneously, 170–180W each — genuine parallel work,
where layer split leaves one card idle while the other runs its half.

**2b. `-sm graph` is the mode that decays with depth.**
Graph runs 199 → 198 → 180 → 153 from pp2048 to pp16384 (−23%), while every
layer implementation is flat or *rising* (buun: 178 → 199) and tensor is nearly
flat. This is the cost of `ik`, and it lands hardest exactly where agentic
transcripts live.

**3. `ik`'s own `-sm layer` is broken, and it poisoned the first read.**
80.08 at pp16384 versus 187.87–198.67 for the other three engines on the
identical model and flags — a 2.3× deficit that no other engine shows. The
initial "graph is +91% over layer" figure was measured against this defect,
not against a competent baseline. Against a good layer implementation the
result is a **trade**, not a win. Corrected here deliberately: the within-ik
comparison is the misleading one, and it was the first one run.

**4. The three non-ik engines are interchangeable on layer split.**
pflash / buun / mainline sit within 5% of each other at every depth
(pp16384: 187.87 / 198.67 / 188.69). Engine choice on layer split is not a
performance decision — so it can be made on features (DFlash2, arch support)
instead.

**5. The AllReduce backend is a correctness switch, not a performance one.**
`internal` and `none` are indistinguishable — every cell within 0.2%, most
within 0.05%, against stddevs of 0.02–0.16, and both runs took exactly 608s.
No fallback warning was logged, so `internal` really did initialise; AllReduce
volume simply isn't a bottleneck here. What matters is avoiding `nccl`, which
aborts.

**Operational consequence:** `-sm tensor` on mainline **requires**
`GGML_CUDA_ALLREDUCE=internal` (or `none`) on this rig. NCCL is the Linux
default and it aborts in 5s. Bake this into every mainline invocation.

**6. The rebase onto current master helps tensor split, and only tensor split.**
`pr-27342` replayed onto `75844307` (115 upstream commits) and re-measured:

| | pp2048 | pp4096 | pp8192 | pp16384 | tg128 |
|---|---|---|---|---|---|
| tensor | **+3.6%** | **+2.0%** | **+1.1%** | **+1.0%** | −0.0% |
| layer | +0.3% | −0.1% | −0.0% | −0.8% | +0.1% |

The tensor deltas are 10–100× their stddevs (±0.02–0.14), so they are real.
Decode is untouched in both modes.

Most likely `d9b6be07`, which moved cuBLAS handles from one-per-device to
one-per-device-**per-stream** and dropped the explicit `cublasSetStream` calls.
Per-stream handles can only pay where concurrent streams exist: tensor split
runs both cards at 97–99% simultaneously, layer split serialises them. That
also explains why "it's the cuBLAS prefill path" is *not* sufficient — layer
prefill is cuBLAS GEMM too and gained nothing. Inference from the diff plus
these two cells, not instrumented.

**Still missing from this phase:** the single-GPU (`none`) leg, which needs
`UD-IQ3_S` to fit in 16 GB.

---

## Phase 2 — drafters on `-sm tensor`

`mainline-rebased` (`57affa09`, PR #27342 on `75844307`) · `UD-Q4_K_M` ·
`llama-server` · `-ngl 99 -fa 1 -t 8` · `REPS=3` · `N_PREDICT=400` ·
`GGML_CUDA_ALLREDUCE=internal`. Rows are means of 3 reps.

| Depth | Drafter | Decode t/s | Prefill t/s | Acceptance | VRAM (0/1) |
|---|---|---|---|---|---|
| 4k | none | 20.30 | 68.0 | — | 8183 / 8183 |
| 4k | MTP-Q4_0 | 21.27 | 59.1 | 40.1% | 9827 / 9827 |
| 16k | none | 19.75 | 199.0 | — | 8579 / 8579 |
| **16k** | **MTP-Q4_0** | **29.26** | **198.4** | **73.3%** | 10235 / 10235 |

**MTP's speedup is a function of context depth: +4.8% at 4k, +48.2% at 16k.**
Acceptance nearly doubles over the same span (40.1% → 73.3%) on an identical
drafter and an identical 400-token budget. This is the same effect the H11 depth
check found for DFlash2 (45.8% → 64.8%), but larger.

**At depth the drafter is free on prefill** — 198.4 vs 199.0 t/s, inside noise.
At 4k it costs 13% (68.0 → 59.1). The drafter's fixed per-request setup is
invisible once a ~15k-token prompt amortises it.

`-sm tensor` splits VRAM exactly evenly in every cell, which layer split never
does (compare the H11 rows: 8647 / 10997 and similar).

**29.26 t/s at 16k is the best decode in the project**, against Phase 1's best of
22.05 (`ik` `-sm graph`, no drafter, tg128) and layer + DFlash2 at 16k at 17.08.

### The matched 2x2 at 16k — split mode vs drafter

Same drafter, same prompt, same 400-token budget, `REPS=3`. This isolates the
two effects against each other at the depth the agentic phase will run at.

| Split | Control | + MTP | MTP gain | Prefill cost of MTP | Acceptance |
|---|---|---|---|---|---|
| layer | 12.35 (183.9 pp) | 19.05 (162.9 pp) | +54.3% | **−11.4%** | 77.5% |
| **tensor** | 19.75 (199.0 pp) | **29.26 (198.4 pp)** | +48.2% | **−0.3%** | 73.3% |

Four things fall out of this:

1. **Split mode is the larger effect.** Tensor beats layer by +59.9% on the
   drafter-free control, and by +53.6% with MTP on both sides. The drafter is
   worth ~50% on top of whichever split you pick; the split is worth ~55–60%
   regardless of drafter.
2. **MTP pays slightly *less* on tensor** (+48.2% vs +54.3%), the same direction
   as at 4k (+4.8% vs +10.3%) but a far smaller gap. Tensor split makes each
   decode step cheaper, so there is less for speculation to reclaim.
3. **The drafter is free on prefill under tensor and expensive under layer** —
   −0.3% vs −11.4%. This is the clearest advantage tensor has beyond raw
   throughput, and it is not something the decode numbers show.
4. **Acceptance is consistently lower on tensor** (73.3% vs 77.5% here; 40.1% vs
   44.4% at 4k). Reduction-order differences perturb the logits. Tensor wins
   anyway, by enough that the acceptance penalty is irrelevant.

**Best measured configuration: `-sm tensor` + MTP-Q4_0 at depth, 29.26 t/s.**
Adding `GGML_CUDA_P2P=1` should put it near 30 (H10: +2.9% on decode), untested
in combination.

### Why the smoke test was wrong by 13x

The gating smoke run reported MTP at 32.60 t/s and **90.0%** acceptance — a
claimed +61.9%. It used `N_PREDICT=64`. The first 64 tokens of a response are its
most formulaic stretch, which is exactly what a small drafter predicts well; the
same drafter on the same prompt run to 400 tokens accepts 40.1%. Short
generations flatter drafters badly, and acceptance measured over one is not a
usable number.

### Acceptance is not a pure property of drafter and task

Same drafter, same prompt, same token budget, 4k: layer split accepts 44.4%
(`draft_n` 513), tensor split accepts 40.1% (`draft_n` 543). Tensor split changes
matmul reduction order, so logits differ slightly and drafts diverge. The effect
is small next to the depth effect, but it means split mode has to be held fixed
when comparing acceptance rates.

### DFlash2 is excluded from this matrix by necessity

It aborts on load under `-sm tensor` in the meta-backend split planner
(`ggml-backend-meta.cpp:543`, `handle_per_row` given an axis-0-split source).
The source is the target's `output.weight`, which DFlash2 borrows via
`cparams.ctx_other` instead of shipping; tensor split shards it on axis 0. MTP
ships its own `output.weight` and `token_embd.weight`, so its graph never sees a
split tensor. Structural — see H8. DFlash2 is layer-split-only on this rig.

### Preliminary — not part of the matrix

One-prompt load check, **not** `llama-bench`, **not** comparable to the rows
above or to Phase 1. Recorded because it is the first evidence DFlash2 runs here.

| Engine | Quant | Drafter | Split | Decode | Acceptance | Notes |
|---|---|---|---|---|---|---|
| mainline | UD-Q6_K_M | DFlash2-Q4_K_M | layer | 16.10 t/s | 239/419 = 57.0% | `-c 4096`, 300 tok, temp 0, one prompt. Load 4m40s, 13.8/15.0 GiB, 49°C |

---

## Phase 3 — quant sweep

**Not started.** The model-quant sweep below is still pending.

| Engine | Split | Quant | pp16384 | tg128 | VRAM | Notes |
|---|---|---|---|---|---|---|
| _pending_ | | `IQ3_S` | | | | Run on both `layer` and `none` — the only single-GPU-capable target |
| _pending_ | | `Q4_K_M` | | | | |
| _pending_ | | `Q5_K_M` | | | | |
| _pending_ | | `Q6_K_M` | | | | |

### KV-cache quant sweep on `buun` — partial (2026-08-25)

Ran ahead of Phase 3 to answer whether TurboQuant justifies switching engines.
`buun` · `-sm tensor` · `UD-Q4_K_M` · MTP-Q4_0 · 16k depth · `ALLREDUCE=none`.

| `-ctk`/`-ctv` | Decode t/s | Prefill t/s | Acceptance | VRAM (0+1) | Reps |
|---|---|---|---|---|---|
| `f16`/`f16` | **28.12** | 178.2 | 82.4% | 21552 MiB | 3 |
| `turbo3`/`f16` | 24.76 | 177.7 | 74.2% | 21160 MiB | 1 |
| `turbo3`/`turbo3` | 24.09 | 177.8 | ~77% | 20764 MiB | 3 |
| `turbo3`/`q8_0` | — | — | — | — | **not run — stopped** |

**TurboQuant is a loss on this rig: −14.3% decode for 788 MiB saved at 16k.**
The 788 MiB is far less than an f16→3-bit KV conversion should yield at this
depth, which suggests turbo is not applied to every layer's cache — unconfirmed,
because `buun`'s server log emits no KV sizing lines. Answers H4 and H7.

`turbo3/q8_0` was killed mid-run when work was stopped; its three
`FAILED(request)` rows were removed from `results/h11-placement.csv` as kill
artefacts rather than measurements.

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
| H4 (TurboQuant KV penalty) | `results/h25-iq3-1card.csv` | **Does not generalise.** The 14.3% figure was TurboQuant's own; stock `-ctk/-ctv q8_0` costs **1.5%** of decode and 0.2% of prefill. See Phase 7 below |
| H5 (NCCL harmful) | `phase0-ik-graph-q6k` | **Partly confirmed** — fails outright, not merely slower. See HYPOTHESES.md |
| H7 (dispatch bug) | _pending_ | |
| H8 (DFlash2 usable) | `h8-loadcheck-df2q4` — runs on both GPUs, 57.0% acceptance, clean output | **Works.** Speedup not yet quantified — needs mainline's own no-drafter control |
| H9 (Q8 vs Q4 drafter) | `h11-df2q8-*` vs `h11-df2q4-*` | **Leaning no** — Q8 accepted 44.4% vs Q4's 45.8%, no faster, +0.87 GiB. One prompt; needs more |
| H11 (drafter placement) | `results/h11-placement.csv` | **Refuted** — ≤0.15% spread across placements. Found a `-devd CUDA0` abort with DFlash2 instead |


---

## Phase 7 — prefill, TTFT and the single-GPU question

### H24 — does `-ub 2048` cost decode? (2026-08-27)

Two matched llama-server arms, 16k context, MTP drafter, `-sm tensor`,
`GGML_CUDA_ALLREDUCE=none`, `-b 2048`, 400 output tokens, 3 reps.
Data: `results/h11-placement.csv` (`h24-ub512-mtp-16k`, `h24-ub2048-mtp-16k`).

| `-ub` | decode t/s | prefill t/s | acceptance | VRAM/card |
|---|---|---|---|---|
| 512 | 29.39 | 198.5 | 73.3% | 10,235 MiB |
| 2048 | 27.41 | 310.5 | 66.4% | 10,973 MiB |
| delta | **-6.7%** | **+56.4%** | -6.9 pts | +738 MiB |

**Refuted, and explained exactly.** The acceptance control moved, which normally
voids a test — here the arithmetic closes it with no residual. 400 output tokens
cost `400 - accepted` target forward passes: 126 vs 135 steps is +7.1% work,
predicting -6.7% decode against -6.7% measured. **There is no intrinsic ubatch
penalty on the decode path**; the loss is entirely the drafter accepting less at
the larger ubatch. Generated tokens were byte-identical across arms and reps, so
quality is untouched.

**Verdict: keep `-ub 2048`.** Break-even is ~11,800 output tokens per prompt —
far beyond any real request, and the config is for long-running harnesses where
TTFT is what hurts.

Side benefit: the 29.26 t/s headline reproduced at 29.39 under
`ALLREDUCE=none`, closing a config-change gap in the record.

### H25 — 27B-IQ3_S on one P100 (2026-08-27)

Three single-card arms, `-dev CUDA0 -sm none`, `-ub 2048`, <=16k.
Data: `results/h25-iq3-1card.csv`.

**KV quantisation is nearly free** (the durable result, and it is about the
*two-card* config):

| KV | pp16384 t/s | tg128 t/s |
|---|---|---|
| f16 | 183.80 | 11.38 |
| q8_0 | 184.13 | 11.21 |
| q4_0 | 184.05 | 11.18 |

Prefill within 0.2%, decode within 1.5%. `q4_0` buys nothing over `q8_0`.
**Consequence for the deliverable:** `q8_0` should halve the two-card 100k KV
cache from ~6,250 to ~3,125 MiB for ~1.5% of decode.

**One card vs the pair, at 16k:**

| | one card (IQ3_S) | pair (Q4_K_M) | ratio |
|---|---|---|---|
| prefill | 184.1 | 327.1 | 56% |
| decode, no drafter | 10.4 | 19.75 | 53% |
| decode, as actually served (MTP) | **OOM** | 27.41 | **38%** |

Halving the weight bytes did not buy back the lost card: decode fell to the same
53% as prefill. MTP does not fit — the server loaded at 16,029 of 16,269 MiB and
died on the first request, already with `q8_0` KV, so the usual mitigation was
spent:

```
common_fit_params: failed to fit params to free device memory:
  n_gpu_layers already set by user to 99, abort
CUDA error: out of memory ... in ggml_cuda_pool_vmm::alloc
```

**Verdict (user call, 2026-08-27): a single P100 is not feasible for real-world
use, and the line of enquiry is closed.** It fails on speed and on the 100k
context ceiling before quality is even measured. The offered follow-up (MTP at
`-ub 512`) was declined: its best case still loses to the two-card
drafter-free number while also surrendering prefill.
