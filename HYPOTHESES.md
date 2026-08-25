# Open hypotheses

Each is a claim to verify, mostly carried from
`/root/p100_inference_knowledge_handover.md`, which predates the current engine
builds. **Treat every one as unproven until a run in this repo confirms it.**

Update the Status line when a run bears on a hypothesis, and reference the
run label so the evidence is traceable.

| ID | Claim | Status |
|---|---|---|
| H1 | MTP wins bigger on P100 than on a 3090 | Untested |
| H2 | XL quants stall one GPU on the split | Untested |
| H3 | Stock quants below Q6 degrade on `ssm_out` | Untested |
| H4 | TurboQuant KV avoids the q4_0 KV penalty | Untested |
| H5 | NCCL is harmful for split inference | **CONFIRMED — not a slowdown, an abort** |
| H6 | `-sm graph` is the best split mode | **Partly confirmed — a trade: +76% decode, -23% deep prefill.** `-sm tensor` cell still open |
| H7 | `(turbo*, F16)` KV combo aborts | Untested |
| H8 | DFlash2 is worth using on this rig | **Runs on both GPUs, 57% acceptance** — speedup unquantified |
| H9 | Q8 drafter beats Q4 by more than it costs | Untested — no longer gated |
| H10 | Inter-GPU transport (P2P / AllReduce backend) is leaving performance on the table | Untested |
| H11 | Drafter placement across the two cards matters | **REFUTED at 4k and 16k** — placement moves VRAM, not throughput |
| H12 | `ik`'s graph-split tuning knobs (`-smf16`, `-gap`, `-smgs`, `-sas`) change the `-sm graph` verdict | Untested (unblocked — target: graph's -23% prefill decay) |

---

## H1 — MTP win size

**Claim:** MTP should give a *larger* relative speedup on P100 than the ~1.7×
reported for Qwen3.6-27B on a 3090 (38 → 65 t/s).

**Reasoning:** MTP trades compute for bandwidth — N draft tokens verified per
weight read. The P100 is severely bandwidth-bound and has a 2:1 FP16 rate that
the batched verification path exercises well.

**Test:** tg128 with vs without MTP, same engine/quant/split. Record acceptance
rate alongside throughput — a speedup without a plausible acceptance rate is a
measurement error, not a win.

**Status:** Untested. Requires Phase 2. MTP draft head is on disk and verified
(`nextn_predict_layers = 1`).

---

## H2 — XL quant split-stall

**Claim:** `-XL` quants upcast the embedding and output head to Q8_0. Those
tensors can't be split, so they land entirely on one GPU and stall the other
during prefill. XL is claimed to be *strictly better* on a single GPU.

**Why it matters:** determines whether XL quants are usable at all in the
dual-GPU configuration, which is the whole point of this rig.

**Test:** `Qwen3.8-27B-UD-Q4_K_XL.gguf` on `-sm layer` and `-sm row` with
telemetry running. Look for one device sitting at low utilization during the
prefill phase while the other saturates. Compare against `UD-Q4_K_M` on the
same split as the control.

The evidence should be numeric — per-GPU utilization from
`logs/<label>.temps.log`, not an impression. Explicitly requested as
"proof, not trust me."

**Status:** Untested. Model on disk.

---

## H3 — Stock quant degradation below Q6

**Claim:** Non-dynamic ("stock") quants below Q6 degrade disproportionately on
Gated Delta Network / `ssm_out` layers. Unsloth Dynamic quants are claimed
exempt.

**Test:** `Qwen3.8-27B-Uncensored-noMTP-Q4_K_M.gguf` (stock) vs
`Qwen3.8-27B-UD-Q4_K_M.gguf` (dynamic). **This is a quality comparison, not a
speed one** — throughput will be near-identical and is not the question.

Needs a coherence/perplexity measure, not `llama-bench`. Define the measure
before running, and note that the two models differ in more than quantizer
(one is an "Uncensored" fine-tune), so this is suggestive rather than clean.

**Status:** Untested. Confounded comparison — see caveat above.

---

## H4 — TurboQuant KV penalty

**Claim (open question, not an assertion):** f16 KV was previously mandatory on
these cards — q4_0 KV cost 31–58%. But TurboQuant computes directly on
quantized values rather than dequantizing per access, so it may not inherit
that penalty.

**Test:** same engine/quant/split, sweeping `-ctk`/`-ctv` across `f16`,
`q8_0`, and the turbo types. Watch for the H7 dispatch bug when choosing
combinations.

**Status:** Untested.

---

## H5 — NCCL harmful for split inference

**Claim:** NCCL is actively harmful (−14% on a MoE model) for `ik_llama`'s
split path. No NVLink means direct CUDA peer-to-peer beats AllReduce over
PCIe. Original guidance: "do not build with NCCL."

**Status: CONFIRMED, and the reality is worse than the claim.** NCCL doesn't
cost 14% — it makes `-sm graph` unusable.

Proven with a matched pair in Phase 1: same tree, same commit (`8337e4c`),
same model, same flags, **one CMake flag apart**.

| Build | Result |
|---|---|
| `build/` (`GGML_NCCL=ON`, the CMake default) | **exit 134 after 9s** — aborts on the first prefill test |
| `build-nonccl/` (`GGML_NCCL=OFF`) | exit 0, 676s, full result set, peak 68°C |

```
ggml_cuda_op_reduce: ncclAllReduce failed with status 1
/root/ik_llama.cpp/ggml/src/ggml-cuda/reduce.cu:169: Fatal error
```

Verified the two builds actually differ as intended: `build/bin/llama-bench`
links `libnccl.so.2` (via `libggml.so`); `build-nonccl/bin/llama-bench` links
neither.

The upstream author agrees, in `ggml/src/ggml-cuda/reduce.cu:146`:

```cpp
#ifdef GGML_USE_NCCL
    // Somehow I'm not able to figure out how to use NCCL correctly.
    // Hence, if enabled, we use NCCL only for the cases where it works and performs well.
```

**Operative guidance — stronger than the original claim:** the −14% figure is
untestable on this rig, because NCCL never produces a number to compare. Build
`ik_llama` with `-DGGML_NCCL=OFF`. `GGML_NCCL` defaults to **ON**, so the
default build is the broken one.

Evidence: `logs/phase1-driver.log` (`phase1-ik-graph-q4km` vs
`phase1-iknonccl-graph-q4km`), `logs/phase0-ik-graph-q6k.log`.

**Not applicable to other engines.** `pflash`/`buun`/`mainline` don't offer
graph split and don't link NCCL; H5 is an `ik` finding only.

---

## H6 — Best split mode overall

**Claim:** `ik_llama`'s `-sm graph` was the prior known-good baseline for this
rig (on the much older build `286ce324`).

**Status: partly confirmed — graph is a trade, not a win.** Phase 1 settled
the no-MTP half on `UD-Q4_K_M`.

| | pp2048 | pp16384 | tg128 |
|---|---|---|---|
| ik `graph` | **199.14** | 153.04 | **22.05** |
| ik `layer` | 116.49 | 80.08 | 13.54 |
| pflash `layer` | 184.31 | 187.87 | 12.53 |
| buun `layer` | 178.47 | **198.67** | 13.10 |
| mainline `layer` | 180.38 | 188.69 | 12.88 |

**Graph owns decode.** 22.05 vs 12.53–13.54 — every layer implementation in
every engine sits in a 1.0 t/s band, and graph is 63–76% clear of all of them.

**Graph loses deep prefill.** It is the only mode that decays with depth
(199 → 153, −23% out to 16k) while the others stay flat or rise. Crossover
is near 6k.

**Correction on the record:** the first reading of this ("graph +91% over
layer") compared graph to *ik's own* layer, which Phase 1 shows is anomalously
broken — 80.08 at pp16384 against 187.87–198.67 for three other engines on
identical work. Comparing within one engine was the wrong control; the
cross-engine comparison is the honest one, and it turns a blowout into a trade.

**Consequence for engine choice:** pflash / buun / mainline are within 5% of
each other at every depth, so layer-split engine choice is a *feature*
decision, not a performance one. The only thing `ik` uniquely buys is decode
throughput via graph — at the cost of deep prefill, an NCCL-disabled custom
build (H5), and an older feature set.

**Still open:**
1. **`-sm tensor` on mainline** — the decisive cell. `-sm tensor` was
   previously recorded here as unusable for `qwen35`; **that was wrong**.
   `llm_arch_supports_sm_tensor()` is a *denylist* (`default: return true`)
   and `qwen35` is not on it, on our built branch as well as upstream tip. The
   binary at `/root/dflash2-llama.cpp/build-cuda-p100/bin/llama-bench` already
   advertises `<none|layer|row|tensor>`. No rebuild needed. If tensor reaches
   graph's decode, the case for `ik` collapses.
2. `ik`'s `-sm attn` (`LLAMA_SPLIT_MODE_ATTN = 2`) — reachable from
   `llama-cli`/`llama-server` but not `llama-bench`. Untested.
3. Phase 2 re-tests the combination with MTP, which is the comparison that
   actually decides the rig's configuration.

---

## H7 — Known FATTN dispatch bug

**Claim:** the `FATTN_VEC_CASE` dispatch table has `(F16, turbo*)` but not the
reverse `(turbo*, F16)`. So asymmetric KV types abort:

```bash
--cache-type-k turbo3 --cache-type-v f16     # aborts at fattn.cu:348
--cache-type-k turbo3 --cache-type-v turbo3  # works
--cache-type-k turbo3 --cache-type-v q8_0    # works
```

**Test:** confirm present or absent on each of the three current trees before
relying on any asymmetric turbo KV configuration. This is a cheap check — the
abort happens early, so it doesn't need a full benchmark run.

**Status:** Untested on current builds. The claim was made against a different
fork and may not apply to all three.


---

## H8 — DFlash2 is worth using on this rig

**Context:** DFlash v1 and PFlash were excluded from testing after the operator
found both too lossy on these cards — v1 also regularly mangled tool calls.
**DFlash2 is a different drafter** (separate checkpoint, separate upstream PR,
adds a candidate selector v1 lacks), so that verdict is being re-opened for it
specifically. The v1/PFlash exclusion still stands.

**Claim:** DFlash2 gives a materially better decode speedup than the MTP head on
Qwen3.8-27B, without breaking tool calls.

**Status: it runs, on both cards, and the output is clean.** The earlier "no
engine supports it" verdict was right about the three forks and wrong to stop
there — upstream `ggml-org/llama.cpp` **PR #27342** is the engine, it builds for
`sm_60`, and it works. Now the fourth engine `mainline` (METHODOLOGY §3).

### First result — `h8-loadcheck-df2q4` (2026-08-24)

`mainline` · `UD-Q6_K_M` · `DFlash2-Q4_K_M` · `-sm layer` · `-c 4096` ·
`--spec-draft-n-max 7` · one prompt, 300 tokens, `temperature 0`.

| | |
|---|---|
| Load time | 4m40s |
| VRAM | 13.8 GiB (GPU0) / 15.0 GiB (GPU1) — **both cards** |
| Decode | **16.10 t/s** |
| Draft tokens / accepted | 419 / 239 — **57.0% acceptance** |
| Output | Correct iterative linked-list reversal, no mangling |
| Peak temp | 49°C |

**This is a load check, not a benchmark.** One prompt, short context, greedy
sampling. Do not quote 16.10 t/s as the DFlash2 figure.

The tempting comparison — 16.10 vs the Phase 1 `llama-bench` tg128 of 9.45 t/s
(pflash/layer/Q6_K_M, no drafter) — is **confounded**: different engine,
different tool, different context depth. It hints at ~1.7× but proves nothing.
The honest measurement is mainline's own no-drafter control, which is the next
run.

**The dual-GPU question is settled empirically.** 13.8/15.0 GiB resident across
both cards while generating at 57% acceptance. The `buun` tree-verify
restriction does not apply to mainline, as the code reading predicted. The
single-GPU `IQ3_S` leg is still worth running as a quant and single-GPU
reference, but it is no longer a DFlash2 rescue path.

**Verifying v2 turned out to be easier than expected.** The startup log prints
the shared dflash parameters (`n_max=7, block_size=8, n_extract=5`) but not the
v1/v2 branch. It doesn't need to: the response `timings` object carries
`draft_n` and `draft_n_accepted` directly, so acceptance is measurable per
request. `scripts/web_bench_metrics.py` already records the whole timings object,
so Phase 5 captures acceptance rate for free — no log parsing, and no reliance on
a version banner.

### Why the forks can't do this

| Engine | DFlash support | Verdict |
|---|---|---|
| `pflash` | none (`--spec-type` offers only `mtp`, `ngram-*`) | Can't |
| `buun` | DFlash **v1** (`draft-dflash`) + DSpark; docs cite PR #22105 | Wrong version |
| `ik` | `dflash`, `dspark` | Wrong version (same lineage) |
| `mainline` | PR #27342 — v1 **and** v2, auto-detected | ✅ |

The v1-silently-runs risk identified earlier is now concrete and confirmed. PR
#27342 adds **no new `--spec-type` value**: the same `draft-dflash` selects v1 or
v2, and `common/speculative.cpp` branches on draft metadata —

```cpp
selector_top_k = llama_model_dflash_selector_top_k(model_dft);
is_dflash2     = selector_top_k > 0;
```

Our drafters carry `dflash.selector_top_k = 16`, so mainline takes the v2 path.
A v1 loader reads no such key, takes the v1 path, and the command line looks
identical. **Confirm from the log that the selector loaded before recording any
number as DFlash2.**

### The multi-GPU concern does not apply to mainline

This was flagged as a possible blocker; it isn't. The GPU-0-only restriction is
`buun`-specific — its `parent_ids_gpu` **tree** verification, auto-disabled at
`n_devices() > 1` (`src/llama-context.cpp:3343`). Mainline has no tree-verify
path at all: `parent_ids` does not appear anywhere in its `src/` or `common/`,
and PR #27342's selector emits a **linear** draft, walking a predecessor chain
one token at a time and `push_back`-ing a flat sequence. That is verified the
ordinary way and carries no device restriction. The only `n_devices() > 1` guard
in mainline's context is the unrelated pipeline-parallelism check.

So DFlash2 should work on `-sm layer` across both cards. **Still verify it
empirically** — that is a code reading, not a run.

### Test

1. ~~Load check on `mainline` with `-sm layer`~~ — **done**, see above.
2. Phase 2-style decode comparison: none / MTP / DFlash2-Q4 / DFlash2-Q8, same
   target and split, acceptance rate recorded alongside t/s.
3. Phase 5 web-bench for the tool-calling stress test — where v1's mangling
   showed up, and the reason this hypothesis matters beyond throughput.
4. **Single-GPU leg** (operator's request): if dual-GPU DFlash2 misbehaves
   despite the above, re-run on one card with `UD-IQ3_S` and `-sm none`. That
   quant exists precisely to make this possible.

**Note:** `mainline` is a different engine from the three in Phases 1–3, so its
numbers are not directly comparable to theirs. Run the none-drafter control
*on mainline* as the baseline for H8/H9 — comparing DFlash2-on-mainline against
MTP-on-buun would confound drafter with engine.

---

## H9 — Drafter quantisation: Q8 vs Q4

**Claim:** the Q8_0 DFlash2 drafter accepts enough more tokens than the Q4_K_M
one to be worth its extra ~0.9 GB of VRAM.

**This is a speed and VRAM question, not a quality one.** The target verifies
every drafted token over its full vocabulary, so a worse drafter lowers
acceptance and therefore throughput — it cannot change what the target emits.
None of the three engines exposes a lenient-acceptance mode (checked their
flags; `--spec-draft-p-min` gates drafting, not acceptance).

**Test:** identical engine/target/split, three drafter arms plus a control —
none, MTP, DFlash2-Q4, DFlash2-Q8. Record acceptance rate alongside decode t/s;
a speedup without a plausible acceptance rate is a measurement error.

The VRAM side matters here specifically: at `UD-Q6_K_M` (21.5 GiB of a 32 GiB
pool) the drafter competes with KV cache for what's left, so the comparison is
decode t/s *at equal context*, not decode t/s alone.

**Status:** Untested. **No longer gated** — both drafters are on disk, verified
as genuine v2, and `mainline` can load them. Phase 5 is where it gets
stress-tested against real tool-calling work, which is where DFlash v1's mangling
showed up and where a bad drafter will show up again.

Note the VRAM figures are smaller than first assumed: 1.06 GiB (Q4_K_M) vs
1.92 GiB (Q8_0), a 0.86 GiB difference. Beside a 21.5 GiB target in a 32 GiB
pool that is a real but modest KV trade; beside an 11 GiB `IQ3_S` target on a
single 16 GiB card it is the difference that may decide whether the config fits
at all.

---

## H10 — Inter-GPU transport is leaving performance on the table

**Claim:** the default inter-GPU data path is not the best one available on this
rig, and two runtime switches can change it without a rebuild.

**Two specific findings behind this** (mainline, read from
`ggml/src/ggml-cuda/ggml-cuda.cu`):

1. **Peer access is off by default.** `cudaDeviceEnablePeerAccess` is called only
   when `GGML_CUDA_P2P` is set in the environment. These cards *can* peer —
   verified with a direct `cudaDeviceCanAccessPeer` probe: `1` in both
   directions, PHB topology, single NUMA node. So every cross-GPU copy currently
   staged through host memory could plausibly go card-to-card instead.

2. **The AllReduce backend is selectable at runtime.**

   ```cpp
   const char * env = getenv("GGML_CUDA_ALLREDUCE");
   if (!env) { /* Linux default */ ggml_backend_cuda_comm_init_nccl(ret); }
   // else: "nccl" | "internal" | "none"
   ```

   `internal` is a **dedicated two-device pipeline** — it warns
   `internal AllReduce init failed (n_devices != 2?)` and falls back, meaning
   the fast path is written for exactly this rig's shape. `none` is a
   meta-backend butterfly.

**This is H5's other half.** H5 established that NCCL AllReduce *fails outright*
on `ik` (`ncclAllReduce failed with status 1`) and required a rebuild with
`-DGGML_NCCL=OFF`. Mainline is different in two ways worth recording:
`GGML_CUDA_NCCL:BOOL=ON` was auto-detected in our build (NCCL **is** linked, and
is the Linux default), **but** it is switchable per-run by env var rather than by
rebuild. The `h8-loadcheck-df2q4` run succeeded on `-sm layer` because layer
split needs no AllReduce — the NCCL path is not exercised there. A `-sm row`
run on mainline may well hit the same failure `ik` did.

**Test:**

1. `-sm row` on mainline with `GGML_CUDA_ALLREDUCE` unset (NCCL, default) —
   does it reproduce `ik`'s abort? Cheap and settles whether H5 is
   engine-specific or rig-wide.
2. Same, with `GGML_CUDA_ALLREDUCE=internal`, then `=none`. Three-way.
3. `GGML_CUDA_P2P=1` vs unset on `-sm layer` at fixed everything else. This one
   applies to the *current* working config, so it is the highest-value cheap
   test in this hypothesis.

Record `nvidia-smi` per-GPU utilization alongside throughput — the mechanism
claim is about transfer stalls, so utilization is the evidence, not just t/s.

---

## H11 — Drafter placement matters

**Claim:** where the draft model lives changes decode throughput on a two-card
split.

**Reasoning:** with `-sm layer` the target's layers are spread across both cards,
so every draft/verify cycle crosses the PCIe link. The drafter is small
(1.06–1.92 GiB) and could sit entirely on one card via `-devd`, or be placed
per-tensor with `-otd`. Whether pinning it helps (locality, no split overhead) or
hurts (contention with the target's layers on that card, and it is then remote
from half the target) is genuinely unclear.

This interacts with H9: the Q8 drafter is 0.86 GiB larger, so placement and
quantisation trade against each other for VRAM on whichever card holds it.

**Test:** fixed target/split/context, DFlash2-Q4, varying only placement —
default (split), `-devd CUDA0`, `-devd CUDA1`. Record decode t/s **and
acceptance rate**; placement should not change acceptance, so a change there
means something else moved.

**Status: REFUTED** (2026-08-24, `results/h11-placement.csv`).

Ran all three drafters x three placements plus a no-drafter control, 3 reps each.
Decode spread within a drafter is <=0.15% — 14.11/14.12/14.11 t/s for MTP,
14.46/14.48 for DFlash2-Q4. Acceptance was identical across placements of the
same drafter down to the exact draft-token count, confirming placement was the
only thing that moved. VRAM did shift between cards (GPU0 8647-10291 MiB), so
the flag works; decode simply does not care.

Re-tested at depth (`-c 16384`, 14.5k-token prompt), since `-c 4096` with a
short prompt only decodes at ~450 tokens and placement is a locality question:
`default` 17.10 t/s vs `-devd CUDA1` 17.06 t/s, acceptance identical at
263/406 in both. **The null holds at both depths.**

Consequence: `-devd`/`-otd` are **not** a tuning knob on this rig and are
dropped from the Phase 5 matrix. **Use `default` — do not pass `-devd` at all.**
Pinning is not merely neutral, it costs headroom on the fuller card (at 16k,
`CUDA1` leaves 4382 MiB free on GPU1 vs 5182 for `default`) and the penalty
grows with context. `CUDA1` also has no compatibility argument: `default` runs
every drafter too, and only `CUDA0` aborts.

Two things fell out that H11 was not looking for:

- **`-devd CUDA0` aborts with any DFlash2 drafter** at load. The drafter GGUF
  has no `output.weight`/`token_embd.weight` and borrows the target's via
  `ctx_other`; under `-sm layer` that tensor is on CUDA1, so restricting the
  draft context to CUDA0 makes it unreachable and ggml aborts. Upstream bug in
  PR #27342. MTP is immune — it ships its own copies. See RESULTS.md Phase 6.
- **H9 evidence:** the Q8 drafter accepted *less* than the Q4 (44.4% vs 45.8%)
  for +0.87 GiB. Recorded there rather than here.

---

## H12 — `ik`'s graph-split knobs change the `-sm graph` verdict

**Claim:** `-sm graph` was the prior known-good baseline for this rig, and `ik`
ships tuning flags for it that have never been exercised:

| Flag | Default | Hypothesis |
|---|---|---|
| `-smf16` / `-smf32` / `-grt` | f16 | Inter-GPU exchange precision. P100 has native 2:1 FP16 and no NVLink, so f16 exchange should already be the right default — but `-grt` allows other types and this has never been checked |
| `-gap` | f16 | Flash-attn precision under graph split |
| `-smgs` | 0 | Force graph scheduling |
| `-sas` | 0 | Async compute-graph evaluation — the most likely to matter for overlapping PCIe transfer with compute |

**Status: unblocked, untested.** The `-DGGML_NCCL=OFF` rebuild exists at
`/root/ik_llama.cpp/build-nonccl/` and Phase 1 confirmed `-sm graph` runs there
(676s, exit 0). These knobs are now reachable.

Phase 1 also gave them a specific target: graph's prefill decays with depth
(199 -> 153 from pp2048 to pp16384, -23%) while every layer implementation
stays flat. That decay is the thing to attack — `-sas` (async compute-graph
evaluation, overlapping PCIe transfer with compute) is the most plausible lever
if the decay is transfer-bound.
