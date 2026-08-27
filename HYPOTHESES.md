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
| H4 | TurboQuant KV avoids the q4_0 KV penalty | **REFUTED — turbo3/turbo3 costs 14.3% of decode to save 788 MiB. `buun` has no remaining advantage** |
| H5 | NCCL is harmful for split inference | **CONFIRMED and rig-wide — an abort, in both `ik` and mainline** |
| H6 | `-sm graph` is the best split mode | **REFUTED — `-sm tensor` on `mainline-rebased` wins. `buun`'s tensor is 12–14% slower on prefill** |
| H7 | `(turbo*, F16)` KV combo aborts | **REFUTED — `turbo3/f16` runs clean on `buun`. Turbo KV is legal and ~12% slower than f16** |
| H8 | DFlash2 is worth using on this rig | **REFUTED as a configuration — layer-split-only, and `tensor + MTP` beats `layer + DFlash2` by +71% decode at 16k** |
| H9 | Q8 drafter beats Q4 by more than it costs | **Provisionally REFUTED — matched DFlash2 pair is 0.1% apart on decode; Q8 costs 872 MiB and accepts less** |
| H10 | Inter-GPU transport (P2P / AllReduce backend) is leaving performance on the table | **CLOSED — AllReduce: no lever (only butterfly runs on Pascal). P2P: real but small, +2.9% decode / −0.3% prefill** |
| H11 | Drafter placement across the two cards matters | **REFUTED at 4k and 16k** — placement moves VRAM, not throughput |
| H12 | `ik`'s graph-split tuning knobs (`-smf16`, `-gap`, `-smgs`, `-sas`) change the `-sm graph` verdict | Untested (unblocked — target: graph's -23% prefill decay) |
| H13 | 27B prefill holds ≥150 t/s at 100k — it decays less than the 9B did, because 49 of 65 layers are linear-cost | Untested — **run first** |
| H14 | `-ub` 512→1024/2048 improves prefill ≥10% | **CONFIRMED 2026-08-25 — +63%.** Use `-ub 2048` |
| H15 | Lifting the 175 W cap to 220 W improves prefill ≥15% and decode <3% | Untested — **approved to 220 W** |
| H16 | ~~TurboPrefill nets a TTFT win at 64k~~ | **Withdrawn 2026-08-25** — user call. Costs 35% of decode |
| H17 | The sm_60 FP16 fast-path fix is throughput-free and changes model output | Untested |
| H18 | Gated-delta-net layers, not GEMMs, dominate prefill | Untested — **decides the plan** |
| H19 | Prompt-cache reuse cuts agentic turn-2+ TTFT by >90% | Untested |
| H20 | A 9B on one P100 beats the 27B's TTFT by ≥2× at 64k at acceptable NIAH quality | Untested |
| H21 | ~~PFlash-style token selection ports to sm_60~~ | **Withdrawn 2026-08-25** — see below |
| H22 | The existing chunked GDN graph path beats the fused sequential kernel on prefill | Untested — **the last kernel-level lever** |

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

**Status: REFUTED.** TurboQuant does not avoid a KV penalty — it has one of its
own, and a larger one than the question anticipated.

`buun` (the only non-excluded tree with turbo types) · `-sm tensor` ·
`UD-Q4_K_M` · MTP-Q4_0 · 16k depth:

| `-ctk`/`-ctv` | Decode t/s | vs f16 | VRAM (0+1) | Reps |
|---|---|---|---|---|
| `f16`/`f16` | **28.12** | — | 21552 MiB | 3 |
| `turbo3`/`f16` | 24.76 | −11.9% | 21160 MiB | 1 |
| `turbo3`/`turbo3` | 24.09 | **−14.3%** | 20764 MiB | 3 |

**−14.3% of decode to save 788 MiB at 16k context.** That is a bad trade at any
depth this rig can reach, and it gets worse rather than better as context grows,
because the throughput cost scales while the saving evidently does not.

The 788 MiB is itself suspicious: an f16→3-bit KV conversion at 16k should free
far more. The likely explanation is that turbo is not applied to every layer's
cache, but `buun` emits no KV sizing lines in its server log, so this is
unconfirmed. It does not change the verdict — the decode cost is measured
directly and is what decides it.

**Consequence: no reason to run `buun` over `mainline-rebased`.** TurboQuant was
the only capability `buun` had that the rebased tree lacks (see METHODOLOGY's
engine capability matrix), and it is not worth having. `buun` also trails on
prefill by 12–14% and on decode by 3.9% with MTP attached.

`turbo3`/`q8_0` was queued but never ran — work was stopped first. Given the
trend across the three cells above it is unlikely to change the verdict, but it
is genuinely untested.

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

### `buun` also has `-sm tensor`, and it is slower (2026-08-25)

Tensor split is not unique to the rebased tree. `buun` offers it and the arch
gate is the same denylist (`src/llama-arch.cpp:1023`, `default: return true`),
so `qwen35` is permitted there too. It runs — and loses:

| test | mainline rebased | buun | delta |
|---|---|---|---|
| pp2048 | 222.75 | 193.50 | −13.1% |
| pp4096 | 223.43 | 191.86 | −14.1% |
| pp8192 | 214.88 | 188.57 | −12.2% |
| pp16384 | 208.25 | 182.75 | −12.3% |
| tg128 | 20.34 | 20.67 | +1.6% |

A 12–14% prefill deficit for 1.6% of decode. Too large to be the per-stream
cuBLAS commit alone (measured at +1.0–3.6% in the rebase validation), so
`buun`'s base is materially older than current master in the prefill path.

With a drafter attached at 16k the decode advantage reverses as well: `buun` +
MTP gives 28.12 t/s against the rebased tree's 29.26 (−3.9%), at 178.2 vs 198.4
prefill (−10.2%).

**So H6's answer is not just "tensor" but "tensor on `mainline-rebased`".**

⚠️ Acceptance rates are **not** comparable across engines. `buun` reports 82.4%
against the rebased tree's 73.3% on the same drafter, prompt, and token budget,
while drafting fewer tokens (`draft_n` 301 vs 374) — its speculative defaults
differ. End-to-end decode t/s is the fair comparison; acceptance is not, unless
`n_max`/`p_min` are pinned on both.


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

**Status: REFUTED on `buun` — the only tree it could be tested on.**

Turbo KV types exist only in `buun` and `pflash` (`pflash` is excluded from
testing). Neither `ik` nor `mainline` has them, so `buun` is the whole
population here.

`buun` · `-sm tensor` · `UD-Q4_K_M` · MTP-Q4_0 · 16k depth · 1 rep:

```
-ctk turbo3 -ctv f16   ->  runs clean, 24.76 t/s, 177.68 pp, 74.2% acceptance
```

No abort, no `fattn.cu:348`. The claim was inherited from a different fork and
the caveat recorded against it ("may not apply to all three") was correct.

**The useful finding is not the refutation — it is that turbo KV is slower.**
Against `f16/f16` on the same engine and depth (28.12 t/s), `turbo3/f16` costs
**12% of decode**. That is the opposite of what H4 hoped for, and it is measured
on the asymmetric combination this hypothesis said was impossible.

Caveat: the H7 cell ran 1 rep, because it was built as an abort check rather
than a measurement. The sign is unambiguous, the magnitude is provisional.


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

**Status: REFUTED as a configuration, at 16k depth.** DFlash2 does beat MTP
head-to-head on layer split — but that contest no longer decides anything,
because it cannot follow MTP onto tensor split.

| Configuration | Decode t/s | Prefill t/s | Acceptance |
|---|---|---|---|
| **tensor + MTP** | **29.26** | 198.4 | 73.3% |
| layer + MTP | 19.05 | 162.9 | 77.5% |
| layer + DFlash2-Q4 | 17.08 | 159.7 | 64.8% |
| layer control | 12.35 | 183.9 | — |

**`tensor + MTP` beats `layer + DFlash2` by +71.3% on decode and +24.2% on
prefill.** No amount of DFlash2's drafter quality closes a gap created by the
split mode it cannot use.

The original H8 claim — "DFlash2 gives a materially better decode speedup than
the MTP head" — is not what was tested here and remains unmeasured on equal
footing, because equal footing does not exist: there is no split mode both can
use where either is competitive. On layer at 16k, DFlash2 is *behind* MTP
(17.08 vs 19.05), which is the reverse of the claim, though acceptance differs
enough (64.8% vs 77.5%) that spec-parameter defaults may account for it.

**What stands from H8:** DFlash2 runs, on both cards, with clean output, and the
"no engine supports it" verdict was wrong. What falls: any reason to use it.

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

**Finding 4 — P2P is a real but small decode lever. H10 is now CLOSED.**

`mainline-rebased` · `UD-Q4_K_M` · `-sm tensor` · `GGML_CUDA_ALLREDUCE=none` ·
`-r 3`. Only `GGML_CUDA_P2P` differs.

| test | p2p=off | p2p=on | delta |
|---|---|---|---|
| pp2048 | 222.75 ±0.04 | 221.91 ±0.16 | −0.38% |
| pp4096 | 223.43 ±0.11 | 222.77 ±0.17 | −0.29% |
| pp8192 | 214.88 ±0.03 | 214.19 ±0.06 | −0.32% |
| pp16384 | 208.25 ±0.01 | 207.63 ±0.04 | −0.30% |
| **tg128** | 20.34 ±0.05 | **20.92 ±0.04** | **+2.86%** |

Peer access **costs ~0.3% of prefill and buys ~2.9% of decode**. Both effects are
small, but both are outside the noise: the prefill loss is consistent in sign and
size across all four depths against stddevs of 0.01–0.17, and the decode gain is
12× the stddev.

The asymmetry fits the mechanism. Decode is latency-bound — many small cross-GPU
exchanges per token — so removing the host-memory bounce from each one pays.
Prefill is throughput-bound on large batched transfers, where the staged path is
already efficient and enabling peering just adds mapping overhead.

**Verdict: set `GGML_CUDA_P2P=1` for decode-dominated work** (the agentic phase),
leave it off for prefill-dominated work. On a mixed workload it is close to a
wash, tilted positive.

The `p2p=off` cell also reproduced the earlier `phase1-rebased-tensor-q4km` run
to within 0.06% on all five cells across separate invocations — a useful check
on harness repeatability, independent of the P2P question.

**Not confirmed: that P2P stacks with MTP.** Both levers were measured alone —
P2P on a drafter-free `llama-bench` tg128, MTP on a `llama-server` run at depth.
A cell combining them (`p2-tensor-mtp-16k-p2p`) was queued twice and ran
neither time: the first attempt died to a harness defect (a script edited while
that job was executing it), the second was stopped before it started. **Do not
quote ~30 t/s for tensor + MTP + P2P** — the measured figure remains 29.26
without P2P, and the +2.9% is measured in a different regime.

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

---

# Prefill / TTFT hypotheses (H13–H21)

Added 2026-08-25 from `Research/prefill-ttft-2026-08-25.md`, which carries the
reasoning, the source citations and the arithmetic behind these. Read it first —
several obvious-looking prefill knobs are already closed at the source level and
are listed in RUNBOOK's closed table rather than here.

**Framing that applies to all nine.** Prefill is compute-bound at roughly
`2 × 27e9 = 54 GFLOP/token`. Two P100s peak at 37.4 TFLOPS FP16, so a 100k
prefill cannot go below **144 s** however good the kernels get, and 45–55% of
peak — a good outcome — is **260–320 s**. We measure 11.2 TFLOP/s today, 30% of
peak. So H14–H16 are worth at most a ~1.5–2× improvement between them, and only
H19 and H20 can produce a step change, because only those two reduce the amount
of work done. (H21 was a third such lever; it was withdrawn on 2026-08-25.)

---

## H13 — the prefill decay curve to 100k

**Claim:** 27B prefill holds ≥150 t/s at 100k, decaying less steeply than the
Qwen3.5-9B did in the July NIAH runs, because `full_attention_interval = 4`
means only ~16 of 65 layers carry a quadratic term.

**Why it comes first:** we have never prefilled this model past 16,384 tokens.
Every 64k/100k figure in circulation is an extrapolation, and the honest bracket
is wide enough to change the plan: **7.8 min (linear) to 15.9 min (9B decay
shape)** for 100k. Which end we are on decides whether the target is "shave 30%"
or "restructure the work".

**Test:** `llama-bench -p 16384,32768,65536,100000 -n 0` on `mainline-rebased`,
`-sm tensor`, `UD-Q4_K_M`, f16 KV, `GGML_CUDA_ALLREDUCE=none`. Single rep is
fine at the long tiers; they are slow and the variance on prefill has been <1%.
VRAM at 100k is ~22.8 GB of 31.8 GB, so it fits (§2 of the research doc).

**Watch:** the 83 °C limit. A 100k prefill is ~8–16 minutes of sustained
compute, far longer than any run this repo has done. Do not leave it unmonitored.

### Result — CONFIRMED, but the flat-curve assumption behind it was wrong (2026-08-26)

`scripts/run-ubatch-sweep.sh`, `UB=2048`, `-p 16384,65536,100000 -n 0 -r 1`,
`mainline-rebased 57affa09`, `-sm tensor`, `UD-Q4_K_M`, f16 KV,
`ALLREDUCE=none`. 1743 s, peak 70 °C. `results/raw/h13-prefill-depth-q4km.csv`.

| depth | t/s | TTFT |
|---|---|---|
| 16k | 327.1 | 50 s |
| 64k | 250.1 | 4.4 min |
| 100k | **215.4** | **7.7 min** |

The numeric claim holds (215.4 ≥ 150 t/s), but the shape it was framed against
does not: prefill is **not** flat with depth at `-ub 2048` (−34% from 16k to
100k). H14's +63% gain at 2–8k has largely **evaporated by 100k** — the real
100k TTFT (7.7 min) is close to the *best case* of the old `-ub 512` bracket
(7.8–15.9 min) that H14 was supposed to beat. Whether the decay is shallower
than the 9B July NIAH shape was not re-checked — deprioritized in favor of
finding out *why* it decays (see H23, opened from this result).

---

## H14 — ubatch is mis-tuned for prefill

**Claim:** raising `-ub` from 512 to 1024 or 2048 improves prefill by ≥10%.

**Reasoning:** every prefill number in this repo was taken at the default
`-ub 512` / `-b 2048`; it has never been swept. Larger ubatches give cuBLAS
taller GEMMs, and on a card with no tensor cores, GEMM efficiency is strongly
shape-dependent. Two ceilings to respect: `ssm-scan.cu` defines
`SSM_SSD_MAX_TOKENS = 8192`, above which the SSD path falls back to the slow
scan; and activation memory grows with ubatch, which competes with a 6.25 GiB
KV cache at 100k.

**Counter-argument, and why the test is still cheap:** if H18 is right and
prefill is bound by the sequential GDN kernel rather than by GEMMs, ubatch will
move almost nothing — the GDN kernel's cost is linear in tokens regardless of how
they are grouped. **A null result here is positive evidence for H18.**

**Test:** `-ub 512 / 1024 / 2048` × `-b 2048 / 4096`, at 16k, `-sm tensor`.

### Result — CONFIRMED, and by a much larger margin than claimed (2026-08-25)

Two runs, `scripts/run-ubatch-sweep.sh`, `mainline-rebased 57affa09`, `-sm tensor`,
`UD-Q4_K_M`, f16 KV, `ALLREDUCE=none`, prefill only, r=2. Raw:
`results/raw/h14-ubatch-sweep-q4km.csv`, `results/raw/h14-ubatch-sweep-hi-q4km.csv`.

Sweep 1 — `-b 2048`, so `-ub` is capped at 2048:

| `-ub` | 128 | 256 | **512 (old default)** | 1024 | 2048 |
|---|---|---|---|---|---|
| p=2048 t/s | 197.0 | 249.7 | **218.9** | 278.8 | **357.5** |
| p=4096 t/s | 194.8 | 241.7 | **221.8** | 276.5 | **351.1** |

Sweep 2 — `-b 8192`, p=8192, to find the top of the curve:

| `-ub` | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|
| t/s | 271.9 | 342.9 | **347.5** | **OOM** |

**Verdict: +63% at 2–4k, +57% at 8k, from one flag.** The claim was ≥10%.

**Where it tops out.** The curve plateaus at 2048–4096 (+1.3% for 2× the
activation VRAM) and `-ub 8192` aborts in `ggml_cuda_pool_vmm::alloc` inside
`ggml_cuda_mul_mat_cublas_impl` — VRAM exhaustion on the dequant→cuBLAS staging
buffer, not the `SSM_SSD_MAX_TOKENS = 8192` fallback that was the predicted
ceiling. **`-ub 2048` is the pick**: 4096's extra 1.3% is not worth doubling
activation memory that has to coexist with a 6.25 GiB KV cache at 100k.

**The curve is non-monotonic, and that is a real finding, not noise.** 256 (249.7)
beats 512 (218.9) at both prompt lengths, with stddev under 0.15 t/s. A smooth
"bigger GEMMs are better" story does not produce a dip at exactly the default. It
points at a shape- or alignment-sensitive path — something in the tensor-split or
GDN launch geometry that 512 lands badly on. Worth understanding before trusting
any single ubatch as optimal at a depth we have not measured.

**What this does to H18.** H14 was framed so a null result would be positive
evidence that prefill is GDN-bound rather than GEMM-bound. We got the opposite of
a null result, so that inference does not fire — a 63% swing from ubatch alone is
GEMM-shaped behaviour. H18 is **not** settled by this (the GDN kernel is still
sequential and still carries its author's TODO), but it is no longer the leading
explanation for the 30%-of-peak figure, and it should be re-run at `-ub 2048`
rather than at the old default.

**What this does to the rest of the repo.** Every prefill number ever recorded
here was taken at `-ub 512` — the one setting in the sweep that underperforms its
neighbours in both directions. All of them are ~35% low. They remain valid as
*relative* engine comparisons (all were taken at the same setting) but must not be
quoted as this rig's prefill capability. Re-baseline before drawing TTFT
conclusions.

**Not tested:** the effect of `-ub 2048` on **decode** or on MTP acceptance, and
whether +63% survives to 64k–100k, where activation memory competes with KV. Both
belong to H13, which should now run at `-ub 2048`.

---

## H15 — the 175 W power cap is throttling prefill

**Claim:** raising the limit from 175 W to 220 W improves prefill by ≥15% while
moving decode by <3%.

**Reasoning:** both cards run at `175.00 W` against a `250.00 W` default — a 30%
power cut. Prefill is compute-bound and therefore clock-bound; decode is
bandwidth-bound and HBM2 clocks are not power-gated the same way. The asymmetric
prediction is what makes this a real hypothesis rather than a knob-turn: if
decode moves as much as prefill, the model of what is limiting us is wrong.

**Status: approved to 220 W (2026-08-25).** The 175 W setting was conservative,
not a measured limit. The ceiling is now **220 W per card — not 250 W**, and that
number is a hard cap on this hypothesis, not a starting point to negotiate up
from.

**One human gate remains before the first run at >175 W.** Two cards at 220 W is
+90 W over the current draw, and nobody has measured what the PSU is actually
pulling at the wall. The user will check this with a plug-socket power meter,
**manually and in person** — it cannot be read from this host. Do not raise the
cap until that check has been reported back.

**How to run it once cleared:** step **175 → 200 → 220**, one step per run, never
jumping straight to 220. Treat the temperature log as the primary output, not the
throughput — this collides directly with the 83 °C rule, and the cards already
reach 70 °C at 175 W on a 684 s run, which leaves 13 °C of headroom to spend.
Abort the step and stay at the previous cap if any card crosses 80 °C. Record the
cap as a CSV column and restore 175 W afterwards; the setting does not survive a
host reboot, so a reboot silently reverts to 250 W — re-apply the cap on every
boot.

---

## H16 — TurboPrefill on two GPUs — **WITHDRAWN**

**Withdrawn 2026-08-25 by user instruction: "TurboPrefill is a no-go."**

The reasoning it was opened on still stands and is why it was never attractive:
TurboPrefill reports 5.3× on 12× P104-100, but that speedup scales with pipeline
depth and **we have two cards**, so the realistic expectation was 1.3–1.6×. It
requires `-sm layer`, and Phase 2 measured layer+MTP at 19.05 t/s against
tensor+MTP at 29.26 — **it costs 35% of decode to buy prefill.** Against the
project's stated goal of *both* high decode and high prefill at 100k, that is the
wrong trade, and it would also mean running a b10335 build that predates our
DFlash2 rebase and PR #26177.

Do not reopen without a change to the decode requirement.

---

## H17 — the sm_60 FP16 fast-path fix

**Claim:** patching sm_60 out of the FP16 fast path leaves prefill unchanged,
gains ~1.4% decode, and measurably changes model output.

**Reasoning:** [issue #25593](https://github.com/ggml-org/llama.cpp/issues/25593)
— sm_61 has a carve-out from the fast path, sm_60 does not, so FP32 math is
silently truncated. Measured upstream: median KL divergence **0.004962 → 0.000001**,
same-top-token rate **95.00% → 99.89%**. Roughly **1 in 20 greedy tokens flips.**
Fix is three lines adding `cc != 600` beside the existing `!= 610` checks.

**Two reasons this matters beyond quality.** First, **`buun` already carries the
fix** (its PR #80) and our `mainline-rebased` does not — so every *quality*
comparison between them in this repo is confounded, though throughput comparisons
survive, the fix being throughput-neutral. Second, the upstream report that
prefill throughput is *identical* with FP32 compute is strong external evidence
for H18: halving the GEMM rate should be visible if prefill were GEMM-bound.

**Test:** patch, rebuild, re-run one Phase 2 cell for throughput, then a NIAH
tier for quality. Free by hypothesis, so if throughput holds, adopt it.

---

## H18 — GDN layers dominate prefill

**Claim:** in a 16k prefill, more than half of GPU time is spent in
`GATED_DELTA_NET`, not in `MUL_MAT`.

**Reasoning:** `full_attention_interval = 4` puts ~49 of 65 layers in the
gated-delta-net path, and that kernel walks tokens one at a time —
`gated_delta_net.cu:63` is `for (int t = 0; t < n_tokens; t++)` with warp
reductions per step and no GEMMs, so it never touches the P100's 2:1 FP16 rate.
Line 180 carries the author's own `//TODO: Add chunked kernel for even faster
pre-fill`; [PR #19504](https://github.com/ggml-org/llama.cpp/pull/19504) shipped
the op as a vector implementation with chunking left as future work. The Mamba-2
path in the same tree *did* get its chunked kernel (`SSM_SSD_CHUNK_SIZE 256`).

Three observations fit: prefill is flat 2k→16k (−6.4%), we sit at 30% of FP16
peak, and upstream reports FP32 compute costs no prefill throughput.

**Update 2026-08-26 — the flat-curve premise breaks at depth.** H13's real
100k measurement shows −34% from 16k to 100k at the same `-ub`, not flat. A
kernel whose cost is linear in tokens (this one) cannot produce *super-linear*
time growth on its own — something else is scaling worse than linearly with
depth. See H23: the arithmetic points at the ~16 full-attention layers'
O(n²) cost, not the GDN kernel, as the better fit for the depth decay
specifically. This doesn't clear H18 — the kernel is still sequential scalar
FP32 with the author's own TODO — but it means H18 and H23 are probably both
true at different depths: GDN-bound short, attention-bound long. Test both
together, at 100k, not 16k.

**Why it decides the plan:** if true, H14 and the whole family of quant/GEMM
knobs are dead ends, and the largest lever we control is writing a chunked GDN
kernel — plausibly **1.5–2.5× on those layers, ~1.3–1.7× end-to-end** (the
published 2–3× figures rely on Hopper TMA and warp specialisation we cannot use;
the chunked *algorithm* itself is architecture-neutral).

**Test, cheapest first:**
1. `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` vs `f16` on a 16k prefill. Halves the GEMM
   rate. If prefill barely moves, prefill is not GEMM-bound. **One run, decisive.**
2. `test-backend-ops perf` filtered to `GATED_DELTA_NET` — **not currently built**;
   needs a rebuild with tests enabled.
3. `nsys`/`ncu` on a single 16k prefill for a per-kernel breakdown.

---

## H19 — prompt-cache reuse for the agentic loop

**Claim:** with the prompt cache sized for the workload, turn-2-onward TTFT in an
agentic session drops by more than 90%.

**Reasoning:** an agentic loop re-sends a nearly identical prefix every turn.
llama-server can keep slot KV in host RAM and restore it — `-cram/--cache-ram`,
`--cache-idle-slots`, `-ctxcp/--ctx-checkpoints`, `-cms/--checkpoint-min-step`.
**The default `--cache-ram` is only 8192 MiB**, and this model's KV is 64 KiB per
token (16 attention layers × 4 KV heads × 256 × 2 × 2 B), so 8 GiB holds roughly
*one* 100k context. We have never tuned this.

**Hard bound:** host RAM is **28 GB total, ~27 GB available** — perhaps three 100k
f16 contexts, fewer with the model's host-side allocations. This is the one place
where KV quantisation might still earn its 14.3% decode cost (H4), by tripling
how many contexts fit in host RAM. Worth a follow-up cell if the bound binds.

**Test:** `--cache-ram 20000 --cache-idle-slots` on the WEB_BENCH agentic
scenario, measuring per-turn TTFT across a multi-turn session against the same
session with `--cache-ram 0`. This is the **highest-value practical change in the
list** for the stated use case, and it costs nothing but a flag.

---

## H20 — a lower-quant 27B on a single P100

**Superseded 2026-08-26 (user call).** Originally framed around Qwen3.5-9B —
withdrawn: a different model isn't a fallback for "serve the 27B", it's a
different deliverable. **Replaced with a same-model fallback:** Qwen3.8-27B
at **IQ3** quantisation, which fits on a single 16 GB card, same as the 9B did.
This keeps the fallback on the actual target model.

**Claim:** Qwen3.8-27B-IQ3 on one P100 delivers ≥2× the two-card `UD-Q4_K_M`'s
prefill at 64k, at NIAH quality acceptable for the target workload.

**Reasoning:** single-card removes all cross-GPU tensor-split traffic, and
IQ3 is roughly half the weight bytes of Q4_K_M, so both the memory-bandwidth
and PCIe-transport terms shrink. The open question is entirely quality — IQ3
is a much more aggressive quant than anything tested here so far, and the
existing quality tooling (NIAH harness) was validated against the 9B, not
against a 27B this low.

**Confirmed on disk (2026-08-26):** `/root/Qwen3.8-27B-UD-IQ3_S.gguf`,
11,483 MiB (11.2 GiB). No quantising needed — this is a re-run, not a build.

**VRAM math, and it changes the depth this fallback can actually reach.** One
P100 is 16,269 MiB. Weights alone leave **4,786 MiB** for KV + activations +
overhead:

| context | f16 KV size | fits in 4,786 MiB? |
|---|---|---|
| 64k | 4,096 MiB | barely — ~690 MiB left for activations, likely too tight at `-ub 2048` (H14 hit VRAM limits with far more headroom on two cards) |
| 100k | 6,250 MiB | **no** — 1,464 MiB over budget at f16 KV |

So as specced, this fallback **cannot reach 100k on one card at f16 KV** —
it tops out somewhere below that, or needs a smaller `-ub`, or needs KV
quantisation. The last option reopens the TurboQuant quality question this
repo already priced at 14.3% of decode on the two-card config — worth
revisiting here since removing KV quant may not be optional this time, not
just a cost/benefit choice.

**Test:** re-run the existing NIAH harness (`/root/niah_test/run_engine.py`,
fixtures already generated) with `CUDA_VISIBLE_DEVICES=0`. First find the
actual achievable depth at f16 KV (start at 32k, step up, watch VRAM — don't
assume 64k fits until measured), then prefill speed at that depth for the
comparison, then the NIAH quality gate at the same depth.

**Watch:** quality is still the deeper question — IQ3 is a much more
aggressive quant than anything tested here, and the NIAH harness was
validated against the 9B, not a 27B this low. Use the multi-needle fixtures
(`niah_*_multi.jsonl`) and WEB_BENCH if this looks promising.

**Not started.**

---

## H21 — speculative prefill without the sm_80 kernels — **WITHDRAWN**

**Withdrawn 2026-08-25, one day after it was opened, by user instruction.**
PFlash requires sm_80 and there is no v2; porting the selection stage by hand is
a build project with no upstream to track. It is not worth thinking about further.

**A correction this repo needs to carry.** When H21 was written it argued that the
earlier "PFlash is too lossy" verdict deserved re-testing because it had been
formed on this rig while the sm_60 arithmetic bug (H17) was live. **That is
wrong.** The PFlash quality work was done on a 7900 XTX, not on these P100s, so
the sm_60 bug never touched it and cannot have confounded it. The argument for
reopening the technique rested entirely on that mistake, and falls with it.

**What survives.** Nothing about the *technique* is disproven — prefilling only
the important spans is still the only surveyed idea with a path below the 144 s
floor, because it is the only one that changes the token count rather than the
rate. It is simply not reachable from any codebase we can build. If a
selection-style prefill ever lands in a llama.cpp-family engine that compiles for
sm_60, reopen it then. Do not hand-port it.

PFlash is now closed in every sense: **the product** (sm_80-only, §8 of the
research doc), **the fork** (`pflash-llama.cpp` stays only as the source of the
built `llama-niah` binary and its fixtures, which remain useful for H13 and H20),
and **the technique** (this hypothesis).

---

## H23 — quadratic full-attention cost, not the GDN kernel, drives the depth decay

**Claim:** at 100k tokens, the ~16 full-attention layers' O(n²) cost is a
larger share of total prefill FLOPs than the linear GDN scan across the other
49 — the opposite of what layer-count share (16/65 ≈ 25%) suggests, and the
better explanation for H13's −34% decay from 16k to 100k.

**Reasoning (arithmetic, not measured — opened from H13's result).** From the
GGUF: `embedding_length = 5120`, `attention.head_count = 24`,
`attention.head_count_kv = 4` (6× GQA), `attention.key_length =
attention.value_length = 256`. Attention width = `head_count × key_length` =
6144. Causal full-attention FLOPs per layer ≈ `2 × n² × 6144`; at n=100,000
that's ≈1.23e14/layer, ×16 layers ≈ **2.0 PFLOP** for attention alone.

The repo's standing "54 GFLOP/token" hardware-floor figure (§ README, § H13)
is the linear `2 × params` approximation. It implicitly treats attention as
negligible — true at the 2–8k depths H14 was measured at, false at 100k where
n (100,000) ≫ d_model (5120). Sanity check against H13's real 100k number: if
the 2.0 PFLOP attention term were the whole bottleneck, achieved rate =
2.0e15 / 464s ≈ 4.3 TFLOP/s ≈ **11% of the 37.4 TFLOPS aggregate peak** — well
below the "31% of peak" figure the linear estimate implies, and consistent
with a term that grows faster than tokens (only quadratic attention can
produce falling t/s from a decoder whose other layers are all O(n)).

**Why it decides the plan, if true.** README currently says sparse attention
"has a low ceiling here" because only 16/65 layers are attention — that
reasoning used *layer count* share. If FLOP share flips at depth (16 layers
could be >90% of prefill FLOPs at 100k, not ~25%), windowed/sparse attention
on just those 16 layers — not a chunked GDN kernel (H22) — is the largest
100k-specific prefill lever, while H18/H22 would still matter for decode and
short-context prefill, where the quadratic term is small.

**Caveat, stated plainly:** this is back-of-envelope arithmetic from GGUF
metadata, done to sanity-check H13's result, not a measurement. It has not
been isolated from the GDN kernel's own contribution, or from VRAM-pressure
confounds at `-ub 2048` + a growing KV cache (H14 was never tested at depth
either). Do not act on it as settled.

**Test, cheapest first:**
1. `nsys`/`ncu` per-kernel timing on a 100k prefill — this was already H18's
   proposed test 3; it now separates GDN-bound from attention-bound time in
   one profile instead of needing two hypotheses' worth of runs.
2. Re-run H14's `f16`/`f32` GEMM-compute-type toggle (test 1) at 100k instead
   of 16k — if prefill barely moves, the bottleneck at depth is not GEMM
   (neither GDN's launch geometry nor attention's matmuls), which would argue
   against this hypothesis.

**Not started.** Opened 2026-08-26, ranked after H18/H22 since it changes
which of those two is worth building.

---

## H22 — the chunked GDN graph path beats the fused sequential kernel

**Claim:** forcing `cparams.fused_gdn_ch = false`, which routes GDN prefill
through the existing `build_delta_net_chunking()` graph instead of the fused
CUDA kernel, improves 8k prefill by ≥20% on sm_60.

**Full analysis: `Research/chunked-gdn-2026-08-25.md`.** The short version, and
the reason this is now cheap:

**The chunked algorithm is already implemented** — not as a kernel, as a graph of
generic ggml ops in `src/models/delta-net-base.cpp` (10 × `ggml_mul_mat`, plus
`solve_tri`/`tri`/`cumsum`, chunk size `CS = 64`). It is the default on every
backend that lacks the fused op, so it is well exercised for correctness. It is
not selected here only because `fused_gdn_ch` defaults true and its probe asks
merely *does the backend support the fused op* — which sm_60 does. **There is no
CLI flag**; forcing it is a ~2-line patch, ideally behind an env var so one binary
can A/B both paths.

**Every op it needs runs on sm_60.** `ggml-cuda.cu:5307-5311` returns `true` for
`CUMSUM`/`TRI`/`DIAG`/`SOLVE_TRI` with **no `cc` comparison**, and none of
`solve_tri.cu`, `tri.cu`, `cumsum.cu` carries a `GGML_CUDA_CC_*` gate.
`solve_tri.cu`'s fast path is `MAX_N_FAST 64`, exactly matching `CS = 64`.

**Why the fused kernel is the suspect.** Its grid is `H × n_seqs × ceil(S_v/4)`
and **contains no `n_tokens`** — 1536 blocks for our model whether the prefill is
512 tokens or 100,000, with tokens as a sequential `for` loop inside each block.
It is scalar FP32 with two serial warp reductions per token and **no GEMM and no
FP16 at all**, so 49 of 65 layers use none of the 18.7 TFLOPS the floor argument
is built on. The recurrence is only ~0.3% of prefill FLOPs, so any material time
share means an efficiency two to three orders of magnitude below the GEMMs.

**H14 raised this hypothesis's value.** The fused kernel is invariant to `-ub`
(same tokens, same per-token cost), so H14's +63% came entirely from the GEMM
side — which by Amdahl makes GDN a *larger* share of what remains. 20% of prefill
at `-ub 512` becomes ~29% at `-ub 2048`; 35% becomes ~47%.

**The strongest reason to think it fails.** The directly analogous Mamba-2 chunked
path is gated `cc >= GGML_CUDA_CC_TURING` (`ssm-scan.cu:829`), introduced whole in
PR #22675 with only the comment *"Requires NVIDIA Turing+ otherwise fallback to
scan"* and **no stated reason**. If that gate encodes a *performance* finding —
that without tensor cores the matmul form loses to the scalar scan — it predicts
this hypothesis fails on Pascal. If it encodes an API constraint (that path stages
`half` for cuBLAS; the GDN graph is FP32 throughout), it does not transfer. Nobody
wrote it down, so this must be measured, not reasoned about.

**Two costs charged against any win.** The unfused path materialises q and k at
3× width, because `H_k = 16 ≠ H_v = 48` and `qwen35.cpp:441` inserts an explicit
`ggml_repeat_4d` that the fused kernel avoids via its GQA fastmodulo. And ~50 ops
per layer instead of 1 means more launches and more intermediates — though those
scale with **ubatch, not context** (~25 MiB per `CS×CS×n_chunks×H_v` tensor at
`-ub 2048`), so they do not grow toward 100k.

**Test:** patch, rebuild for sm_60, then `llama-bench -p 2048,4096,8192 -n 0
-ub 2048 -sm tensor`, fused vs chunked. **Check the load log for CPU-fallback
warnings first** — issue #24712 is exactly the "assigned to CPU ... usually due to
missing support" case, and a silent fallback would look like a refuted hypothesis
rather than a misconfiguration. Decode should be untouched (`n_tokens == 1` takes
the separate autoregressive path) but verify. Gate on one NIAH tier before quoting
a throughput number — the two paths are different arithmetic.

**Either result is worth having.** A win is a large free speedup. A loss explains
the Turing+ gate and closes the last kernel-level lever, leaving only the power cap
and the work-reducing levers.

### Result — FAILED, crashes before running (2026-08-26)

Patched `src/llama-context.cpp`: `if (getenv("LLAMA_GDN_FORCE_CHUNKED")) cparams.fused_gdn_ch = false;`
right after the auto-resolve block (it has to run *after* — auto-resolve
unconditionally overwrites `fused_gdn_ch` from its device-support probe, so
setting it earlier is a no-op). Env-gated, off by default, matches the
codebase's existing debug-flag pattern (same file already has
`LLAMA_GRAPH_REUSE_DISABLE`). Rebuilt `llama-bench` only (incremental, ~1 min).

One run: `LLAMA_GDN_FORCE_CHUNKED=1 llama-bench -ub 2048 -b 2048 -p 16384
-n 0 -r 1 -sm tensor`, same params as H13's 16k measurement for direct
comparison. **Crashed during graph allocation, before the first token:**

```
GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN) failed
  ggml_backend_meta_get_split_state(...)
  ggml_backend_meta_buffer_init_tensor_impl(...)
  ggml_gallocr_alloc_graph(...)
  llama_context::process_ubatch(...)
```

**Verdict: the chunked graph's ops (`cumsum`/`tri`/`solve_tri`/etc.) don't
carry split-axis metadata for `-sm tensor`.** The fused kernel is wired for
tensor-split across two GPUs; the chunked graph, exercised elsewhere only on
backends that don't do tensor-split, isn't. This is a **different failure
than the anticipated one** — it's not that the chunked path is slow or that
Pascal loses without tensor cores (the Turing+-gate worry), it never gets far
enough to measure that. Peak temp 50 °C — aborted in <30 s, no thermal risk.

**Does not settle the underlying question.** The chunked algorithm may still
be faster on sm_60; this only shows it's unwired for our specific split mode.
Two ways forward, neither attempted (user paused further chasing on this):
1. Re-test with `-sm none` (single GPU, no split) or `-sm layer` — if it runs
   there, the win/loss question becomes answerable, just not in the winning
   two-card config, which limits how directly it transfers.
2. Add split-axis annotation for the missing ops — real engineering, not a
   2-line patch anymore; the "cheap to test" framing this hypothesis opened
   with no longer holds.

**Not committed to the fork** (`dflash2-rebased` is a local working copy, not
pushed per this repo's own contribution rules); the 3-line patch is left in
place, inert unless `LLAMA_GDN_FORCE_CHUNKED` is set, for whoever re-attempts
this.

**Ordering:** run **H18 first** (`GGML_CUDA_CUBLAS_COMPUTE_TYPE`, at `-ub 2048`).
H18 sizes the prize; H22 collects it. Running H22 blind risks spending a rebuild
on a 5% slice.
