# Results

**Everything measured on this rig, in one place.** Two harnesses, two tables,
and their numbers are **not interchangeable**:

- **Table A — `llama-bench`**: synthetic, no drafter, no server. For
  engine/split/quant/ubatch comparisons. `pp N` = prefill t/s at prompt length
  N, `tg128` = decode t/s. Generated from `results/raw/*.csv` by
  `python3 scripts/build-matrix.py`; regenerate and paste after any new sweep.
- **Table B — `llama-server`**: real HTTP requests, 400 tokens, temp 0, seed 42,
  mean of the reps. The only harness that measures drafters, acceptance and real
  VRAM. **The serve command is built from this one.** Source:
  `results/h11-placement.csv`, curated by hand.

Table C is the verdict per lever. The reasoning behind each verdict is in
[HYPOTHESES.md](HYPOTHESES.md); what happened on the day is in
[RUNLOG.md](RUNLOG.md); what is still open is RUNBOOK §7.

Constant throughout: 2× Tesla P100-PCIE-16GB (16,269 MiB, sm_60, no NVLink),
**175 W cap** (until 2026-10-15; `p15-<cap>-*` runs at the cap in their label, later runs at 200 W, see `logs/<label>.cap`), `-ngl 99 -fa 1 -t 8`, `GGML_CUDA_ALLREDUCE=none` on tensor
split. Rows marked GPUs=1 used `-dev CUDA0`. `results/all-results.csv` is a
stale aggregate that stops at Phase 1; use it only to grep which labels exist.

---

## Table A — `llama-bench` (no drafter)

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

- **`-ub` is the biggest prefill lever, and 512 is a local minimum.** Down the
  `h14-ubatch-sweep` block: 128 → 197, 256 → 250, **512 → 219**, 1024 → 279,
  2048 → **357**. Every pre-H14 row was taken at 512 and is ~35% low; the
  rankings between those rows still hold. `-ub 4096` needs `-b 8192` for +1.3%;
  `-ub 8192` aborts (backtrace in `h14-ubatch-sweep-hi-q4km.csv`).
- **The +63% does not survive to depth.** `h13`: 327 → 250 → 215 t/s from 16k
  to 100k (−34%). 100k TTFT is 7.7 min. A linear-cost model cannot produce
  falling t/s; the quadratic attention share is the suspect (H23).
- **`-sm tensor` beats `-sm layer` on both axes**: 208 vs 187 prefill, 20.3 vs
  12.9 decode (**+58%**, the largest single-flag effect in the table). Both
  cards run at 97–99% together under tensor; layer leaves one idle.
- **`ik -sm graph` wins decode (22.1) and loses everything else**: 153 at
  pp16384, no tensor split, and the only mode whose prefill decays with depth
  (−23% from 2k to 16k). `ik`'s own `-sm layer` is broken (80 at pp16384).
- **The rebase (`mainline` → `rebased`) helped tensor split only**: +3.6% at
  pp2048, +1.0% at pp16384, 10–100× the stddev; layer and decode unchanged.
  Likely the per-stream cuBLAS handles upstream (`d9b6be07`).
- **H17 explains buun's prefill deficit.** `h17-patched-fp32` (rebased with the
  sm_60 FP16 fix) lands within 0.3% of `buun` at both depths: 193.0 / 182.8 vs
  193.5 / 182.7. The fix costs **12–13% of prefill** and buys +0.36% decode.
  Buun's ~1.2% decode lead is a genuine engine difference. Side effect: only
  ~15% of prefill time is GEMM-rate-bound (H18).
- **P2P**: −0.3% prefill, **+2.9% decode** drafter-free (`h10-p2p-on` vs
  `off`). Never measured with MTP; not in the serve command.
- **KV quant is nearly free** (`h25-iq3-1card-kv-*`): f16/q8_0/q4_0 give
  183.8/184.1/184.0 prefill and 11.38/11.21/11.18 decode. `q4_0` buys nothing
  over `q8_0`.
- `h25-iq3-2card-tensor` is mis-named: it ran `-sm tensor` with `-dev CUDA0`,
  i.e. one card, and matches `-sm none` within 0.4%. **`-sm` is a no-op with
  one device visible.**

## Table B — `llama-server` (real requests, drafters, VRAM)

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
| `saluki-iq2mix-mtp-d16k` | rebased | Saluki IQ2-mix | tensor | 2 | 2048 | f16 | MTP-Q4_0 | 16k | 27.36 | 309.1 | 64.9% | 7075/7075 | 65 C |
| `saluki-iq2mix-abl-embmtp-d16k` | rebased | Saluki IQ2-mix abl. | tensor | 2 | 2048 | f16 | MTP, embedded | 16k | 23.65 | 310.1 | 50.8% | 6915/6915 | 65 C |
| `saluki-iq2mix-1card-mtp-d16k` | rebased | Saluki IQ2-mix | none | **1** | 2048 | f16 | MTP-Q4_0 | 16k | 17.45 | 179.2 | 64.9% | 12815/259 | 65 C |

**Fork builds (`Tom1tk/llama.cpp-Pascal`, 2026-10-14 →).** Each row adds one lever to the row above it in its group (HYPOTHESES H27–H47). 2-card rows: Q4_K_M, `-sm tensor`, `-ub 2048`, P2P on. 1-card rows: IQ3_S, `-sm none`, no drafter, q8_0 KV. Accept/VRAM/peak from `results/h11-placement.csv`. 175 W unless the label says otherwise.

| Run | Lever added | Drafter | KV | Ctx | Decode | Prefill | Accept | VRAM/card | Peak |
|---|---|---|---|---|---|---|---|---|---|
| `h28-2card-mmvh1-mtp-p2p` | half2 matvec (H28) | MTP | f16 | 16k | 37.66 | 314.3 | 68.2% | 11053 | 63 C |
| `serve-fork-16k` | + F16 reduce, ALGO3, Q6_K mmvh (H34/H35) | MTP | f16 | 16k | 39.48 | 353.0 | 72.4% | 11053 | 65 C |
| `h36-duplex-16k` | + duplex reduce (H36) | MTP | f16 | 16k | 40.37 | 365.3 | 72.4% | 11053 | 65 C |
| `h41-wide-16k` | + Q6_K wide (H41b) | MTP | f16 | 16k | 41.51 | 365.4 | 72.8% | 11053 | 65 C |
| `h43-k4scale-16k` | + K-quant scale unpack (H43) | MTP | f16 | 16k | 42.39 | 363.2 | 72.8% | 11053 | 65 C |
| `h44-lowbit-16k` | + LOWBIT on 2 cards (H44) | MTP | f16 | 16k | 44.95 | 363.8 | 72.8% | 11053 | 65 C |
| `h45-head96k-16k` | + 96k-row drafter head (H45) | MTP head96k | f16 | 16k | 46.35 | 365.8 | 72.8% | 11157 | 65 C |
| `p15-175w-16k` | + FA_GEMM (H46) | MTP head96k | f16 | 16k | **47.47** | 376.4 | 75.9% | 11157 | 65 C |
| `p15-200w-16k` | cap 200 W (H15) | MTP head96k | f16 | 16k | 47.55 | 393.5 | 75.9% | 11157 | 66 C |
| `p15-220w-16k` | cap 220 W (H15) | MTP head96k | f16 | 16k | 47.63 | 400.5 | 75.9% | 11157 | 68 C |
| `p15-250w-16k` | cap 250 W (H15, one-off) | MTP head96k | f16 | 16k | 47.61 | 400.8 | 75.9% | 11157 | 69 C |
| `serve-fork-100k` | fork at 100k (pre-H36) | MTP | q8_0 | 98k | 26.94 | 232.0 | 65.3% | 12227 | 66 C |
| `h46-ctrl-100k` | serve command pre-H46 | MTP head96k | q8_0 | 98k | 29.65 | 231.4 | 65.3% | 12331 | 69 C |
| `h46-fagemm-100k` | + FA_GEMM (H46) | MTP head96k | q8_0 | 98k | 28.38 | 275.7 | 60.5% | 12331 | 67 C |
| `p15-200w-98k` | cap 200 W (H15) | MTP head96k | q8_0 | 98k | 28.43 | **287.8** | 60.5% | 12331 | 70 C |
| `h27-1card-graphs0` | 1-card fork baseline | none | q8_0 | 16k | 11.00 | 196.3 | — | 13161 | 65 C |
| `h28-1card-mmvh1` | half2 matvec (H28) | none | q8_0 | 16k | 11.82 | 194.0 | — | 13161 | 65 C |
| `h29-1card-mmvh-iq3` | + IQ3 on mmvh (H29) | none | q8_0 | 16k | 15.86 | 196.7 | — | 13161 | 65 C |
| `h42-1card-iq3-lowbit` | + LOWBIT (H42) | none | q8_0 | 16k | 17.89 | 201.2 | — | 13161 | 65 C |
| `h46-1card-iq3-fagemm` | + FA_GEMM (H46) | none | q8_0 | 16k | **18.02** | 205.2 | — | 13161 | 65 C |
**Reading notes.**

- **`h26-100k-mtp-kvq8` is the deliverable config** and the only row at the
  target depth. Two `FAILED(request)` rows for the same label in the CSV are a
  harness bug (a ~340 KB prompt through argv hit E2BIG, curl posted an empty
  body), fixed 2026-08-27 — not a model or VRAM failure.
- **`_n/a_` prefill in the `h11-*` block is invalid, not missing.**
  `cache_prompt: false` did not defeat slot-level prefix reuse, so reps 2–3
  skipped most of prefill. Decode and acceptance are unaffected. Never quote an
  h11 prefill number.
- **`smoke-tensor-mtp` (32.60 t/s, 90%) is one rep of 64 tokens on a cold
  cache.** The same config over 3 reps of 400 tokens (`p2-tensor-mtp`) is
  21.27 at 40.1%. Short generations flatter drafters by ~13×.
- **Split mode is a larger effect than the drafter.** At 16k, tensor beats layer
  by +60% drafter-free (19.75 vs 12.35) and +54% with MTP (29.26 vs 19.05); MTP
  is worth ~50% on top of either. Under tensor the drafter is free on prefill
  (−0.3%); under layer it costs 11%.
- **Acceptance depends on depth, ubatch and split, and is not monotonic.** MTP:
  40.1% (4k) → 73.3% (16k, `-ub 512`) → 66.4% (16k, `-ub 2048`) → 60.3% (100k).
  Tensor split accepts a few points less than layer (reduction order). No
  drafter conclusion transfers across those without re-measuring.
- **`-ub 2048` costs 6.7% of decode at 16k, entirely via acceptance** (H24):
  126 vs 135 target steps for 400 tokens predicts −6.7% exactly; output was
  byte-identical. Break-even is ~11,800 output tokens, so it stays.
- **16k → 100k costs 27% of decode and 33% of prefill** at the same settings,
  and 6 points of acceptance.
- **DFlash2-Q4 beat MTP at 4k/layer** (14.44 vs 14.04) but has never been run
  at the current config (tensor, `-ub 2048`, 16k+), and cannot be: it aborts
  under `-sm tensor` (H8). `-devd CUDA0` also aborts it at load (bug, H11).
- **Q8 drafters are not worth it** (H9): 44.4% vs Q4's 45.8%, same decode,
  +0.87 GiB.
- **Drafter placement (`-devd`) moves VRAM, not throughput** (≤0.15%, H11).
- **TurboQuant KV** (`p3-buun-*`): −14.3% decode for 788 MiB at 16k. Buun's
  acceptance (81.6%) is not comparable to rebased's (73.3%) — different
  `n_max`/`p_min` defaults, same drafter.
- **One card is not viable** (`h25-iq3-1card-*`): 56% of the pair's prefill,
  53% of its drafter-free decode, MTP OOMs at 16,029/16,269 MiB, cannot reach
  100k. Closed by user call.
- **175 W does the thermal work.** Hottest cell anywhere is 70 °C against an
  83 °C limit, including two consecutive ~8-minute 100k prefills (`h26`).

## Table C — lever verdicts

| Lever | Values tested | Verdict | Evidence |
|---|---|---|---|
| Engine | mainline, rebased, buun, `ik`, PFlash | **rebased** (mainline PR #27342 on current master) | Table A phase1 block, H6 |
| `-sm` | `layer`, `tensor`, `graph` (`ik`), `none` (1 card) | **`tensor`** — +58% decode, +11% prefill over layer | H1/H6 |
| `GGML_CUDA_ALLREDUCE` | nccl (default), `internal`, `none` | **`none`**. nccl aborts rig-wide; `internal` needs sm70 and falls back to the same path | H5, H10 |
| P2P | on, off | +2.9% decode drafter-free; +3.6% with MTP (`d5-p2p-mtp-16k`). **Adopted 2026-10-14** | H10 |
| `-ub` | 128 … 8192 | **2048** — +63% prefill at ≤16k, +56% at 16k on the server; 8192 aborts | H14, H24 |
| `-b` | 2048, 8192 | **2048** — only matters as a ceiling on `-ub` | H14 |
| KV `-ctk`/`-ctv` | f16, q8_0, q4_0, turbo3 | **`f16`** since 2026-10-15 (H48b): 98k decode step 98.4 → 88.7 ms (−9.9%, 28.4 → 32.8 t/s), prefill unchanged, fits at 13,573 MiB/card. Earlier q8_0 verdict was at 16k, before the fork's FA kernels; q4_0 adds nothing; turbo3 costs 14.3% | H25, H4, H48, H48b |
| Model quant | Q4_K_M, Q6_K (PFlash only), IQ3_S (1 card), IQ4_XS | **Q4_K_M**. IQ4_XS failed the KLD gate (0.072 vs 0.059, H53). Phase 8 passed: KLD 0.057 vs Q5_K_M FP32, 19/19 haystack at 100k | H30, H32 |
| Drafter | none, MTP-Q4_0, DFlash2-Q4, DFlash2-Q8 | **MTP**. DFlash2 is layer-only; Q8 drafters not worth it | H8, H9 |
| Draft length `--spec-draft-n-max` | 2, 3, 4 (5 skipped) | **3 (default)**. n=2 −4.5% at 16k, −8.0% at 98k; n=4 −26% (step-time cliff at a 5-token verify batch). H55 found the cliff in mmvh (rows/warp 4 → 2 above 4 columns); fork `MMVH_R4` fixes it but n=4 is still −6.1% (44.66 vs 47.54), so n=3 stays | H49, H55 |
| Drafter placement `-devd` | default, CUDA0, CUDA1 | No effect | H11 |
| sm_60 FP16 fast path | stock vs patched out | **Leave on** (patch costs 12–13% prefill). Quality trade **closed as negligible** by H30: 0.018 KLD, 0.31× the quant step | H17, H30 |
| GPUs | 1, 2 | **2 required** | H20/H25 |
| Power cap | 175, 200, 220, 250 W | **200 W** (user call). Prefill +4.4% at 98k / +4.5% at 16k; 220 W +6.4% at 16k; 250 W no more (draw ≤ 216 W). Decode flat | H15 |
| Fork levers (Tom1tk/llama.cpp-Pascal) | H27–H47 | **Adopted:** MMVH (H28/H29), F16_REDUCE (H31), GEMM_ALGO3 (H33/H35), DUPLEX_REDUCE (H36), Q6K_WIDE (H41b), K-quant scale unpack (H43), MMVH_LOWBIT (H42/H44), drafter head96k (H45), FA_GEMM (H46). **Refuted:** CUDA graphs (H27), internal AllReduce (H37), half2 tile Q·K (H38), parallel dispatch (H40), fp16 GEMM Q·K (H47), MMVH_R4 (H55, −6.1% at n=4), FA_GQA6 (H54: GQA-6 head packing, −3.0% 98k step but −1.4% decode from worse drafting on changed text; 1-card +0.9%). **1-card only:** DEQUANT_STAGED (H39r, +3.9% 1-card prefill at 200 W; 2-card +0.9%, not adopted) | Fork-builds block in Table B |
| Context depth | 2k … **100k end to end** | Serves at 100k with 4,122 MiB/card spare | H13, H26 |
| Sustained thermals | 2 × ~8 min 100k prefill | 70 °C peak. Not a constraint at 175 W | H26 |

Phase 5 (agentic web build, opencode v2, fork at 5542bf9, README serve command with f16 KV and 100k ctx, 200 W).
Scored per [WEB_BENCH.md](WEB_BENCH.md) §4. Raw numbers are in `results/web-bench.csv` and `results/web/p5-fork-opencode.json`.

| | Stage 1 | Stage 2 | Stage 3 | Total |
|---|---|---|---|---|
| Wall time | 364 s | 689 s | 759 s | **2136 s** (35.6 min, includes the 322 s cold load) |
| Requests | 14 | 3 | 10 | 27 |
| Generated tokens | 12237 | 29832 | 27427 | 69496 |
| Prompt-cache hit rate | 93.0% | 99.7% | 99.9% | 98.8% |
| Decode t/s (avg) | 44.9 | 41.4 | 37.5 | 41.8 (31.5–51.9) |
| Prefill t/s (avg) | 127.9 | 65.5 | 67.7 | 98.7 |
| Quality | serves, service active | restyle + working JS widget | `game.js` via `<script src>`, hold-to-reel, catch, score | 3/3, one-shot, 0 malformed tool calls |

MTP acceptance over the run: 66.0% (46149/69960). Hottest card 65 °C. Prefill averages are low because
most requests re-prefill only a few hundred uncached tokens. The run is decode-bound.
