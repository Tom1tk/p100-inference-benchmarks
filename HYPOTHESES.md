# Open hypotheses

Each is a claim to verify, mostly carried from
`/root/p100_inference_knowledge_handover.md`, which predates the current engine
builds. **Treat every one as unproven until a run in this repo confirms it.**

Update the Status line when a run bears on a hypothesis, and reference the
run label so the evidence is traceable.

| ID | Claim | Status |
|---|---|---|
| H1 | MTP wins bigger on P100 than on a 3090 | **REFUTED as stated — and speedup is depth-dependent, not a constant: 1.05× @4k, 1.48× @16k** |
| H2 | XL quants stall one GPU on the split | Untested |
| H3 | Stock quants below Q6 degrade on `ssm_out` | Untested |
| H4 | TurboQuant KV avoids the q4_0 KV penalty | Untested |
| H5 | NCCL is harmful for split inference | **CONFIRMED and rig-wide — an abort, in both `ik` and mainline** |
| H6 | `-sm graph` is the best split mode | **REFUTED — mainline `-sm tensor` wins: fastest prefill at every depth, decode within 8%** |
| H7 | `(turbo*, F16)` KV combo aborts | Untested |
| H8 | DFlash2 is worth using on this rig | **Constrained — aborts under `-sm tensor` (borrowed axis-0-split `output.weight`); layer-split-only, so it must beat MTP-on-tensor from a 37% hole** |
| H9 | Q8 drafter beats Q4 by more than it costs | Untested — no longer gated |
| H10 | Inter-GPU transport (P2P / AllReduce backend) is leaving performance on the table | **AllReduce half CLOSED — only butterfly works on Pascal (`internal` needs sm70+ `__nanosleep`); NCCL aborts.** P2P untested |
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

**Status: REFUTED at the claimed size — and the claim was missing a variable.**

Phase 2, `-sm tensor`, `UD-Q4_K_M`, `REPS=3`, 400 tokens:

| Depth | Control | + MTP | Speedup | Acceptance |
|---|---|---|---|---|
| 4k | 20.30 | 21.27 | **1.05×** | 40.1% |
| 16k | 19.75 | 29.26 | **1.48×** | 73.3% |

Neither figure reaches the ~1.7× reported on a 3090, so the hypothesis as
written is refuted. But the more useful finding is that **"the" speedup does not
exist as a single number — it is a function of context depth**, and the claim
never specified one. The bandwidth-bound reasoning above is not wrong, it is
just not the binding constraint: what moves the speedup is *acceptance*, which
nearly doubles between 4k and 16k on an identical drafter and budget.

That points at a mechanism the original claim didn't consider: a long prompt
constrains the plausible continuation, so a small draft head guesses right far
more often. The same pattern appeared for DFlash2 in the H11 depth check
(45.8% → 64.8%). A 3090 comparison would have to fix depth to mean anything, and
the published 3090 number almost certainly came from a shallow-context run — in
which case the honest comparison at 4k is 1.05× vs 1.7×, and P100 does *worse*
relatively, not better.

Two secondary results recorded here because they bear on how this gets measured:

- **A 64-token generation overstated the speedup by 13x** (claimed +61.9% at
  90.0% acceptance; the truth at 400 tokens was +4.8% at 40.1%). Acceptance
  measured over a short generation is not a usable number.
- **Acceptance shifts with split mode.** Same drafter, prompt, and budget at 4k:
  44.4% on layer, 40.1% on tensor. Tensor split changes matmul reduction order,
  so logits differ slightly and drafts diverge.

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

**Rig-wide, not `ik`-specific.** Superseded the earlier note here that called
this "an `ik` finding only" — mainline's `-sm tensor` aborts the same way, in
`ggml_backend_cuda_comm_allreduce_nccl`, 5s in. Any mode on any engine that
performs a cross-GPU AllReduce fails on this hardware. Layer split is unaffected
because it never calls AllReduce, which is why every `-sm layer` cell in Phase 1
ran fine on NCCL-linked builds.

The engines differ only in the escape route:

| Engine | Mode needing AllReduce | Escape |
|---|---|---|
| `ik` | `-sm graph` | rebuild with `-DGGML_NCCL=OFF` |
| mainline | `-sm tensor` | `GGML_CUDA_ALLREDUCE=internal` or `none` at runtime |

See H10 for the transport measurements.

---

## H6 — Best split mode overall

**Claim:** `ik_llama`'s `-sm graph` was the prior known-good baseline for this
rig (on the much older build `286ce324`).

**Status: REFUTED as stated — `-sm graph` is not the best split mode.**
Mainline's `-sm tensor` is.

| | pp2048 | pp4096 | pp8192 | pp16384 | tg128 |
|---|---|---|---|---|---|
| **mainline `tensor`** | **214.83** | **219.13** | **212.70** | **206.32** | 20.34 |
| ik `graph` | 199.14 | 198.45 | 179.56 | 153.04 | **22.05** |
| buun `layer` | 178.47 | 194.31 | 200.73 | 198.67 | 13.10 |
| mainline `layer` | 180.38 | 190.96 | 192.14 | 188.69 | 12.88 |
| pflash `layer` | 184.31 | 191.64 | 192.33 | 187.87 | 12.53 |
| ik `layer` | 116.49 | 109.50 | 96.99 | 80.08 | 13.54 |

**Tensor split wins prefill at every depth** and holds shape (−6% from its 4k
peak out to 16k, vs graph's −23%), beating graph by **+35% at pp16384**. It
costs **7.8%** of graph's decode. Telemetry confirms real parallelism: both
cards at 97–99% utilisation, 170–180W each.

**What actually separates the modes** is whether the *computation* is split or
merely the *layers*. Graph (22.05) and tensor (20.34) both break out; all five
layer-split cells sit at 12.53–13.54 regardless of fork. Layer split is the
slow thing — the engine is incidental.

**Two corrections on the record.**

1. The first reading ("graph +91% over layer") compared graph to *ik's own*
   layer, which is anomalously broken — 80.08 at pp16384 against 187.87–198.67
   for three other engines on identical work. Wrong control.
2. `-sm tensor` was recorded here as arch-blocked for `qwen35`. Wrong twice
   over: `llm_arch_supports_sm_tensor()` is a **denylist** (`default: return
   true`), so `qwen35` is permitted; and when `-sm tensor` did fail, the cause
   was NCCL (see H10), not the arch gate.

**Consequence — `ik` lock-in is not real.** Mainline gives better prefill
everywhere, decode within 8%, plus DFlash2, current arch support, and a
*runtime* escape from the NCCL bug instead of `ik`'s custom-build requirement.

**Required flag:** `-sm tensor` needs `GGML_CUDA_ALLREDUCE=internal` (or
`none`) on this rig. NCCL is the Linux default and aborts in 5s.

**Still open:**
1. Phase 2 re-tests the combination with MTP — tensor split's interaction with
   speculative decoding is unmeasured and could still move the ranking.
2. `GGML_CUDA_P2P=1` on tensor split (H10 test 3). Now the highest-value cheap
   test in the project: tensor split is communication-heavy, and peer access is
   off by default.
3. `ik`'s `-sm attn` (`LLAMA_SPLIT_MODE_ATTN = 2`) — reachable from
   `llama-cli`/`llama-server` but not `llama-bench`. Lower priority now.

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

### DFlash2 is incompatible with `-sm tensor` (2026-08-25)

Phase 1 selected `-sm tensor` as the best split mode, which makes this a hard
constraint on H8 rather than a footnote: **DFlash2 cannot run in the split mode
we want to use.** It aborts during load:

```
ggml-backend-meta.cpp:543: GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0) failed
```

That assert is in `handle_per_row`, the meta-backend's split planner: a per-row
op was handed a source split along axis 0 — the row axis — which it cannot plan
for. The source in question is the target's `output.weight`, which DFlash2
*borrows* via `cparams.ctx_other` rather than shipping its own. Under `-sm
tensor` that borrowed tensor is sharded across both cards on axis 0, so the
drafter's graph inherits a split it was never designed to see.

This is structural, not a knob. It also explains cleanly why MTP survives the
same test: MTP ships its own `output.weight` and `token_embd.weight`, so its
graph consumes unsplit tensors.

**Consequence: DFlash2 is layer-split-only on this rig.** Since layer split
costs ~37% of decode throughput before any drafter is involved (12.79 vs 20.30
t/s control), DFlash2 now has to beat MTP-on-tensor from a hole that deep. The
H8 comparison is therefore no longer drafter-vs-drafter; it is
`tensor + MTP` vs `layer + DFlash2` as whole configurations.

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

**Status: half settled.** Tests 1 and 2 are done — run on `-sm tensor` rather
than `-sm row`, which is the better target since tensor is the mode that won
Phase 1. Test 3 (P2P) is untested and is now the highest-value cheap test left.

**Finding 1 — H5 is rig-wide, not `ik`-specific.** Mainline's `-sm tensor` on
the NCCL default aborts in 5s, the same way `ik`'s `-sm graph` does:

```
ggml-cuda.cu:106: CUDA error
#4  ggml_backend_cuda_comm_allreduce_nccl(...)
#5  ggml_backend_meta_graph_compute(...)
```

This confirms the prediction recorded above ("a `-sm row` run on mainline may
well hit the same failure"). NCCL AllReduce is broken on this hardware
regardless of engine. The difference is the *escape hatch*: mainline switches
by env var, `ik` needs `-DGGML_NCCL=OFF` at build time.

**Finding 2 — the transport choice does not affect throughput.**

| test | `internal` | `none` | delta |
|---|---|---|---|
| pp2048 | 214.83 ±0.08 | 214.68 ±0.16 | −0.1% |
| pp4096 | 219.13 ±0.13 | 219.12 ±0.11 | −0.0% |
| pp8192 | 212.70 ±0.02 | 212.71 ±0.13 | 0.0% |
| pp16384 | 206.32 ±0.02 | 206.40 ±0.04 | 0.0% |
| tg128 | 20.34 ±0.05 | 20.38 ±0.03 | +0.2% |

Identical to within 0.2%, most cells within 0.05%, against stddevs of
0.02–0.16. Both cells took exactly 608s.

**Finding 3 — and the reason they are identical: `internal` never runs on this
rig.** The two cells above measured the *same code path* twice. Every
tensor-split run on P100 falls back to the meta-backend butterfly, logging:

```
internal AllReduce init failed (n_devices != 2?); falling back to meta-backend butterfly
```

The warning's guess is wrong — we do have exactly 2 devices. The real gate is in
`ggml/src/ggml-cuda/allreduce.cu:405`:

```cpp
// The chunked kernel uses __nanosleep, which is sm70+ (Volta+).
for (size_t i = 0; i < n_devices; ++i) {
    const int cc = ggml_cuda_info().devices[devices[i]].cc;
    if (cc < GGML_CUDA_CC_VOLTA) { ... return nullptr; }
}
```

`GGML_CUDA_CC_VOLTA` is 700 (`common.cuh:52`); P100 is `sm_60`, cc 600. The
internal AllReduce depends on `__nanosleep`, a Volta+ instruction, so it is
**architecturally unavailable on Pascal** — not a tunable, not a config
accident, and no `GGML_CUDA_P2P` setting can unlock it.

I previously recorded that "no `internal AllReduce init failed` warning was
logged, so the dedicated two-device pipeline really did initialise." That was
wrong: those were `llama-bench` runs, and `llama-bench` does not surface backend
warnings. `llama-server` logs it on every start. The source gate is decisive
either way.

**So the H10 claim is wrong as stated on the AllReduce half:** the backend is
not leaving performance on the table, because on this hardware there is only one
backend that works. The full P100 picture:

| `GGML_CUDA_ALLREDUCE` | what actually happens on P100 |
|---|---|
| `nccl` (Linux default) | **aborts** — `ggml_backend_cuda_comm_allreduce_nccl` |
| `internal` | cc gate fails → falls back to butterfly |
| `none` | butterfly |

`internal` and `none` are the same thing here. Prefer `none`: it is honest about
what runs and skips a failed init. Every tensor-split number in this project was
measured on the butterfly path.

**Remaining test — the one that could still pay off:**

`GGML_CUDA_P2P=1` vs unset on `-sm tensor` at fixed everything else. Peer
access is off by default, these cards peer in both directions (verified with
`cudaDeviceCanAccessPeer`), and tensor split moves far more cross-GPU data than
layer split does — so this is a better test bed than the `-sm layer` config
originally proposed here.

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
