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
| H5 | NCCL is harmful for split inference | **Partly confirmed — worse than claimed** |
| H6 | `-sm graph` is the best split mode | Untested (`graph` blocked by H5; single-GPU leg unblocked by IQ3_S) |
| H7 | `(turbo*, F16)` KV combo aborts | Untested |
| H8 | DFlash2 is worth using on this rig | **Runs on both GPUs, 57% acceptance** — speedup unquantified |
| H9 | Q8 drafter beats Q4 by more than it costs | Untested — no longer gated |

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

**Status: partly confirmed, and the reality is worse than the claim.**

On the current build (`8337e4c`), NCCL doesn't merely cost performance — it
**fails outright**:

```
ggml_cuda_op_reduce: ncclAllReduce failed with status 1
/root/ik_llama.cpp/ggml/src/ggml-cuda/reduce.cu:169: Fatal error
```

Aborts on the first prefill test, before producing any throughput number.
`GGML_NCCL` defaults to **ON** in ik_llama's CMake, so this is the default
build's behaviour on this rig.

Evidence: `logs/phase0-ik-graph-q6k.log` (run `phase0-ik-graph-q6k`).

**Remaining work:**
1. Rebuild ik_llama with `-DGGML_NCCL=OFF` and confirm `-sm graph` runs.
2. Quantify the performance claim properly — that needs a working NCCL build
   to compare against, which may not be achievable if NCCL simply doesn't
   function here. If so, record that the −14% figure is untestable on this rig
   and the operative guidance is "NCCL is unusable," which is stronger.
3. Check whether this affects `pflash`/`buun` at all — the original finding was
   specific to `ik`'s graph mode, and those engines don't offer graph split.

---

## H6 — Best split mode overall

**Claim:** `ik_llama`'s `-sm graph` was the prior known-good baseline for this
rig (on the much older build `286ce324`).

**Question:** does graph split still win on current builds once MTP is factored
in? `pflash`/`buun` don't have graph split at all, so if MTP is the bigger
lever, the engine choice may flip.

**Test:** Phase 1 establishes the no-MTP baseline across engines and split
modes; Phase 2 re-tests with MTP. The comparison that matters is the
*combination*, not either axis alone.

**Status:** Untested — `-sm graph` is blocked by H5. `none`/`layer` comparisons
can proceed on all three engines meanwhile.

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
