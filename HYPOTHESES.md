# Open hypotheses

Each is a claim to verify, mostly carried from
`/root/p100_inference_knowledge_handover.md`, which predates the current engine
builds. **Treat every one as unproven until a run in this repo confirms it.**

Update the Status line when a run bears on a hypothesis, and reference the
run label so the evidence is traceable.

| ID | Claim | Status |
|---|---|---|
| H1 | MTP wins bigger on P100 than on a 3090 | **REFUTED as stated — and speedup is depth-dependent, not a constant: 1.05× @4k, 1.48× @16k** |
| H2 | XL quants stall one GPU on the split | **Parked by rule zero** (RUNBOOK §7): Phase 8 showed no Q4_K_M gap to fill |
| H3 | Stock quants below Q6 degrade on `ssm_out` | **Parked by rule zero** (RUNBOOK §7): confounded (different fine-tune); Phase 8 showed no Q4_K_M gap |
| H4 | TurboQuant KV avoids the q4_0 KV penalty | **REFUTED — turbo3/turbo3 costs 14.3% of decode to save 788 MiB. `buun` has no remaining advantage** |
| H5 | NCCL is harmful for split inference | **CONFIRMED and rig-wide — an abort, in both `ik` and mainline** |
| H6 | `-sm graph` is the best split mode | **REFUTED — `-sm tensor` on `mainline-rebased` wins. `buun`'s tensor is 12–14% slower on prefill** |
| H7 | `(turbo*, F16)` KV combo aborts | **REFUTED — `turbo3/f16` runs clean on `buun`. Turbo KV is legal and ~12% slower than f16** |
| H8 | DFlash2 is worth using on this rig | **REFUTED as a configuration — layer-split-only, and `tensor + MTP` beats `layer + DFlash2` by +71% decode at 16k** |
| H9 | Q8 drafter beats Q4 by more than it costs | **Provisionally REFUTED — matched DFlash2 pair is 0.1% apart on decode; Q8 costs 872 MiB and accepts less** |
| H10 | Inter-GPU transport (P2P / AllReduce backend) is leaving performance on the table | **CLOSED — AllReduce: no lever (only butterfly runs on Pascal). P2P: real but small, +2.9% decode / −0.3% prefill** |
| H11 | Drafter placement across the two cards matters | **REFUTED at 4k and 16k** — placement moves VRAM, not throughput |
| H12 | `ik`'s graph-split tuning knobs (`-smf16`, `-gap`, `-smgs`, `-sas`) change the `-sm graph` verdict | **Parked by rule zero** (RUNBOOK §7): `ik` has no tensor split; the serve command is now a mainline fork |
| H13 | 27B prefill holds ≥150 t/s at 100k — it decays less than the 9B did, because 49 of 65 layers are linear-cost | **CONFIRMED 2026-08-26** on the number (215.4 t/s), wrong on the shape (−34% 16k→100k). Fork command: 275.75 at 98k (H46) |
| H14 | `-ub` 512→1024/2048 improves prefill ≥10% | **CONFIRMED 2026-08-25 — +63%.** Use `-ub 2048` |
| H15 | Lifting the 175 W cap to 220 W improves prefill ≥15% and decode <3% | **Partly refuted at 16k** (2026-10-15): prefill +4.5% at 200 W, +6.4% at 220 W, +6.5% at 250 W (saturated); decode +0.3%. 98k: +4.4% at 200 W. **User set 200 W as the running cap** (2026-10-15); 220 W at 98k not run |
| H16 | ~~TurboPrefill nets a TTFT win at 64k~~ | **Withdrawn 2026-08-25** — user call. Costs 35% of decode |
| H17 | The sm_60 FP16 fast-path fix is throughput-free and changes model output | **Measured 2026-08-27 — REFUTED on throughput.** Costs **12-13% of prefill**, gains only **+0.36%** decode (claimed +1.4%). Not free, so not adopted; **parked for a decision** that needs quality data. It also explains ~all of buun's prefill deficit |
| H18 | Gated-delta-net layers, not GEMMs, dominate prefill | **Supported 2026-08-27, numerically, as a free side effect of H17.** Halving the GEMM rate cost only 12-13% of prefill, implying **~15% of prefill time is GEMM-rate-bound and ~85% is not.** Still a diagnostic, not a lever — the prize is real but H22 crashes trying to collect it. **Refuted 2026-10-14 by Phase 0 op timings: GEMMs ~87% of prefill, GDN scan ~6% (`results/phase0/`).** |
| H19 | Prompt-cache reuse cuts agentic turn-2+ TTFT by >90% | **CONFIRMED at defaults** (−95.5% to −97.2%), no flag needed |
| H20 | 27B-IQ3_S on one P100 is a viable single-GPU fallback | **CLOSED 2026-08-27 — not feasible.** 56% of the pair's prefill, 38% of its real decode, MTP OOMs, and it cannot reach 100k. Do not spend runs here |
| H21 | ~~PFlash-style token selection ports to sm_60~~ | **Withdrawn 2026-08-25** — see below |
| H22 | The existing chunked GDN graph path beats the fused sequential kernel on prefill | **Tried 2026-08-26, crashes** under `-sm tensor` (chunked ops carry no split-axis metadata). Now a kernel project, not a flag — **parked by rule zero** |
| H23 | Quadratic attention (16/65 layers) drives the decay to 100k | **Parked by rule zero** — arithmetic only; every sparse-attention kernel is sm_80+, so there is no lever on sm_60 |
| H24 | `-ub 2048` costs decode <5% at 16k | **REFUTED 2026-08-27 — costs 6.7%, but entirely via a 6.9 pp drafter-acceptance drop, not the decode path. Keep `-ub 2048`** (the "output byte-identical" claim was vacuous — see H27 correction) |
| H26 | The deliverable config serves at 100k with `q8_0` KV and MTP | **CONFIRMED in substance, narrowly refuted on the letter, 2026-08-27.** It serves: 20.02 t/s decode, 207.4 prefill, 60.3% acceptance, 12,147 MiB/card, 70 C. Decode is 26.9% below 16k, just outside the 25% claimed |
| H25 | IQ3_S on one P100 vs the pair at <=16k | **CLOSED 2026-08-27 — single GPU not feasible.** 56% prefill / 38% real decode / MTP OOMs. The lasting result is for **two cards**: `q8_0` KV costs 1.5% of decode, not H4's 14.3% |
| H27 | CUDA graphs on sm_60 (fork D1, `GGML_CUDA_PASCAL_GRAPHS=1`) cut decode launch gaps | **REFUTED 2026-10-14.** Graphs engage and replay, output is byte-identical, but decode moves −1.1% (2-card) and −0.5% (1-card). The 2-card idle time is sync, not launches. D1 closed; D3 is the 2-card decode lever |
| H28 | Pascal half2 matvec (fork D2, `GGML_CUDA_PASCAL_MMVH=1`) lifts decode | **CONFIRMED 2026-10-14.** +27% 2-card drafter-free, +32.6% with MTP+P2P (37.66 t/s at 16k), +7.5% 1-card. KLD gate passed. In the serve command since H32 passed |
| H29 | mmvh IQ3_S/IQ3_XXS lifts 1-card decode | **CONFIRMED 2026-10-14.** 15.86 vs 11.00 t/s (+44%), KLD gate passed |
| H30 | Phase 8 tier 1: KLD ladder vs a Q5_K_M FP32 reference splits quality loss into quant, arithmetic (H17) and kernel (mmvh) | **Done** (2026-10-14): H17 alone 0.018 = 0.31× the quant step (0.059), and only +0.003 on top of Q4, so **H17 closed as negligible**; mmvh +1.4% (Q4) / +4.2% (IQ3) over mmvq, so **tier 2 runs the candidate** |
| H31 | F16 AllReduce payload (fork P1, `GGML_PASCAL_F16_REDUCE=1`) lifts 2-card prefill | **Confirmed** (2026-10-14, in H34): prefill +6.8% (337.7 → 360.8), decode flat, KLD ≈ 0 (bit-for-bit top-1) |
| H32 | Phase 8 tier 2: the candidate serve command answers every 100k haystack question the Q5_K_M reference answers | **PASSED** (2026-10-14): reference 18/19, candidate 19/19, a superset. The fork command replaces the serve command |
| H33 | A non-default cublasGemmEx algorithm beats ggml's `DEFAULT_TENSOR_OP` at the prefill GEMM shapes (fork P3 probe) | **Prediction wrong, lever found** (2026-10-14): ALGO3 +3–11% (2-card shapes) / up to +25% (1-card) at n=2048, and 1.7–2.7× at n=512–1024; loses below n=256. End-to-end in H34 |
| H34 | Fork levers on one rebuilt binary: Q6_K in mmvh (decode), GEMM ALGO3 (prefill, H33), F16 reduce (H31) | **Partly confirmed** (2026-10-14): Q6_K mmvh 2-card decode +5.2%, 1-card flat; F16 reduce +6.8% prefill; ALGO3 **failed its KLD gate** (0.0037 / 97.5%) and was not run → H35 |
| H35 | GEMM ALGO3 adds no quality loss over the default FP16 GEMM (vs the FP32 Q5 reference), so it can ship | **Confirmed** (2026-10-14): KLD 0.0619 vs 0.0624 default; 2-card prefill +3.1% (360.7 → 371.8); 1-card only +1.0% (predicted +5–10%) |
| H36 | Overlapping the two directions of each 2-card AllReduce copy (`GGML_PASCAL_DUPLEX_REDUCE=1`) lifts prefill | **Confirmed, adopted** (2026-10-14): bit-identical (KLD −0.000005, 100% top-1); prefill +3.5% (353.0 → 365.3), decode +2.3% (39.48 → 40.37), above the ±1% predicted |
| H37 | One-shot small-message AllReduce (`GGML_CUDA_ALLREDUCE=internal` path, fork D3) lifts 2-card decode | **Refuted** (2026-10-14): identity passed, decode −0.5%, prefill flat. Not adopted |
| H38 | half2 Q·K accumulation in the FA tile kernel (fork P5) lifts prefill within quality | **Refuted on quality** (2026-10-14): KLD mean 0.321, max 29.5, same-top 94.6%. Speed arms not run |
| H39 | Staged, coalesced dequant-to-F16 before the prefill GEMM (`GGML_PASCAL_DEQUANT_STAGED=1`) lifts prefill | **Marginal, parked** (2026-10-14): bit-identical; 1-card +4.5%, 2-card +1.0% (noise). Code kept, off; on the higher-cap re-test list (RUNBOOK §1) |
| H40 | Parallel host dispatch per card (`GGML_PASCAL_PAR_DISPATCH=1`) lifts 2-card decode | **Refuted** (2026-10-14): bit-identical, decode +0.3% (noise); cards kernel-busy 89% of decode. Code kept, off |
| H41 | Q6_K mmvh with 4 lanes per superblock (`GGML_PASCAL_MMVH_Q6K_WIDE=1`) lifts 2-card decode | **Confirmed, adopted** (2026-10-14, as H41b): matvec −40% (~190 → ~330 GB/s); decode +2.8% (40.37 → 41.51), prefill flat. The ub-8 KLD gate (H41) failed on rounding-order *difference* only, kernel diff ~1e-6 relative |
| H42 | Low-bit types (Q3_K, Q2_K, IQ2_S/XS/XXS) on mmvh (`GGML_PASCAL_MMVH_LOWBIT=1`) lift 1-card decode | **Confirmed, adopted for 1-card** (2026-10-14): decode **+13.0%** (15.84 → 17.90), prefill flat; KLD vs FP32 Q5 +2.8% (0.3203 → 0.3294, ±0.044), same-top identical. 2-card parked (rule zero) |
| H43 | Branch-free Q4_K/Q5_K scale unpack in mmvh (fork aea9046, always on) lifts 2-card decode | **Confirmed, adopted** (2026-10-15): output identical; decode **+2.1%** (41.51 → 42.39), prefill flat |
| H44 | `GGML_PASCAL_MMVH_LOWBIT=1` (Q3_K and, from fork 2d3b3bf, IQ4_NL on mmvh) lifts 2-card decode | **Confirmed, adopted** (2026-10-15): output identical; decode **+6.1%** (42.39 → 44.96), prefill flat; KLD 0.1241 vs 0.1261 off |
| H45 | Trimming the MTP drafter's LM head to the first 98,304 vocab rows (`mtp-…-head96k.gguf`, fork loader change) lifts 2-card decode | **Confirmed, adopted** (2026-10-15): output identical; decode **+3.1%** (44.96 → 46.35, 3 reps 46.31/46.38/46.35), acceptance unchanged (72.8%, 273/375), prefill flat, +104 MiB/card |
| H46 | `GGML_PASCAL_FA_GEMM=1`: prefill flash attention via cuBLAS GEMMs (Q·K fp32-accumulate, P·V HGEMM, online softmax per 4096-token KV chunk) instead of the tile kernel lifts prefill at depth | **Confirmed, adopted** (2026-10-15): prefill at 98k **+19.2%** (231.36 → 275.75), 1-card 16k ~+3%; KLD no worse than tile vs a non-FA reference (0.555 vs 0.581); H32 haystack 19/19. Decode −4.3% is changed greedy text drafting worse, not the kernel |
| H47 | `GGML_PASCAL_FA_GEMM_QK16=1`: H46's Q·K GEMM in fp16 compute (Q pre-scaled) lifts prefill at depth within H46's quality | **Refuted** (2026-10-15): FA −29% in micro-bench, but KLD vs the non-FA reference 0.618 > tile's 0.581 and same-top 92.30% < 92.63%. Code reverted (fork d3c1c12) |

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

**Status:** Untested. Confounded comparison — see caveat above. **Scheduled as
the last arm of Sweep Q in [QUALITY-PLAN.md](QUALITY-PLAN.md) §2.5**, behind the
arms that can actually change the serve command, and only if there is appetite —
it differs in fine-tune as well as quantizer, so it can suggest and never prove.

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
`sm_60`, and it works. Now the fourth engine `mainline` (METHODOLOGY §2).

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
**H19** can now produce a step change, because it is the only remaining lever
that reduces the amount of work done. (H21 was a second such lever, withdrawn
2026-08-25; **H20 was a third until it was reframed on 2026-08-26** — a 9B cut
the FLOP count, but a 27B at IQ3 is the same 27e9 parameters and therefore the
same prefill FLOPs. IQ3 buys bytes, not arithmetic, and prefill is
compute-bound. See H20 for what it does and does not buy.)

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

### Investigation — 2026-08-27 (patch located, built, not yet measured)

**The patch is exactly three sites**, all in `ggml/src/ggml-cuda/common.cuh`, and
`buun` (`39d97a876`) already carries all three. Copied verbatim from there rather
than reinvented, so the two engines become comparable on quality as well as
throughput. Saved as `patches/h17-sm60-fp16-fastpath.patch`; built into
`/root/dflash2-rebased/build-h17` as engine `rebased-h17`, leaving the
H26-validated `build-cuda-p100` untouched for a clean A/B.

**"Free by hypothesis" is the part that does not survive contact with this
hardware.** The reason sm_61 has the carve-out is that GP104 runs FP16 at **1/64
rate**, so excluding it costs nothing. **GP100 runs FP16 at 2:1** — 18.7 against
9.3 TFLOPS, the card's headline advantage. This patch therefore moves real work
off a genuinely fast path, which sm_61 never had. **The throughput cost is the
thing under test, not an assumption**, and the original framing above understates
the risk.

**Where it bites.** Five consumers, of which one is on the prefill critical path:

| Site | Path | Effect of the patch |
|---|---|---|
| `ggml-cuda.cu:1626` `ggml_cuda_mul_mat_cublas` | **prefill GEMMs** | cuBLAS compute type F16 -> F32 **for quantized `src0`** — i.e. for our Q4_K_M weights |
| `mmvf.cu:650,741` | decode mat-vec | forces `GGML_PREC_F32` |
| `fattn-tile.cuh:322` | flash-attention tile | FP32 accumulate |
| `vecdotq.cuh`, `mmq-load-tiles.cuh` | quantized dot products | compile-time `FAST_FP16_AVAILABLE` |

### Pre-registered prediction, from data already banked

`buun` carries the fix and `mainline-rebased` does not. Against the two engines'
own `-sm tensor`, `ALLREDUCE=none` rows:

| | buun (has fix) | rebased (no fix) | delta |
|---|---|---|---|
| pp2048 | 193.50 | 222.75 | **-13.1%** |
| tg128 | 20.67 | 20.34 | **+1.6%** |

H17 predicts **+1.4% decode**; buun is **+1.6%** ahead. That is close enough to
be worth stating in advance: **the fix may be most of why buun trails rebased by
12-14% on prefill**, in which case the claim that "throughput comparisons survive,
the fix being throughput-neutral" is **wrong** and this repo has been reading a
13% prefill gap as an engine difference when it is an arithmetic-mode difference.

**So the prediction is: patched rebased lands near 193 t/s on pp2048 and near
20.7 t/s on tg128** — i.e. it converges on buun. If it instead stays near 222,
the buun gap is a real engine difference and the fix is genuinely free.

### It is also the cheapest H18 probe we have

H18 (GDN layers dominate prefill) was parked because no cheap test existed. This
patch **halves the theoretical GEMM rate as a side effect**, which is exactly the
lever H18 needs:

- Prefill barely moves -> prefill is **not** GEMM-bound -> **H18 supported**.
- Prefill drops ~40% -> it is GEMM-bound -> **H18 refuted**.

One build, two answers, and the build costs no GPU time at all.

### Test design

- **Test A — paired throughput A/B**, both binaries back to back in one session
  with identical flags, `-p 2048,16384 -n 128 -r 2`. Paired rather than compared
  against the banked control because the entire adoption decision rests on this
  one delta, and the extra arm costs ~5 minutes.
- **Test B — is the patch actually live**: same short prompt, greedy
  (`temp 0, top_k 1, seed 42`), 200 tokens, both binaries, `diff` the output.
  Expect it to differ; roughly 1 greedy token in 20 should flip.

**Not run: a local KL-divergence measurement.** Upstream already measured it
(0.004962 -> 0.000001, top-token 95.00% -> 99.89%) and a local number would not
change the adoption decision, which turns entirely on the throughput cost. Rule
zero.

**Decision rule, fixed before the numbers arrive:** if prefill loss is under ~2%,
adopt — it is a correctness fix for free. If it costs 13% of prefill, it is a
real trade against the 100k TTFT and the call is the user's, not mine.

### Result — 2026-08-27. **The patch is not free. H17 as written is refuted.**

Paired A/B, same tree, same flags, one session, `-sm tensor`,
`ALLREDUCE=none`, `-b 2048 -ub 512`, `-r 2`. Peaks 69 C / 70 C.

| Metric | control (no fix) | patched (FP32) | delta |
|---|---|---|---|
| pp2048 | 222.72 | **193.01** | **-13.3%** |
| pp16384 | 208.44 | **182.76** | **-12.3%** |
| tg128 | 20.357 | **20.430** | **+0.36%** |

Standard deviations were 0.01-0.14 t/s, so every one of these is far outside
noise.

**Scoring the claim as written:**

| H17 said | Measured | |
|---|---|---|
| "prefill unchanged" | **-13.3% / -12.3%** | **refuted** |
| "gains ~1.4% decode" | **+0.36%** | **refuted** — real, but an order of magnitude short |
| "free by hypothesis, so if throughput holds, adopt it" | throughput does not hold | **the adopt-for-free path is closed** |
| "measurably changes model output" | not yet tested | still open |

### The pre-registered prediction was right, and it reprices a banked result

Predicted before the run: patched rebased converges on buun's 193.50 pp2048.
**Measured 193.008 — a 0.25% miss.**

So **the sm_60 fix accounts for essentially all of buun's 13% prefill deficit.**
This repo has been reading an arithmetic-mode difference as an engine difference
since Phase 3. RESULTS.md's note that buun is "12-14% behind rebased on prefill"
is true only because buun is doing the arithmetic correctly and rebased is not.
**Like for like, the two engines have the same prefill.**

The decode half of the prediction **failed**: predicted +1.6% (buun's lead),
measured +0.36%. buun still leads the patched build by ~1.2% on decode, so
buun's decode advantage is a **real engine difference** — smaller than assumed,
but not explained by this fix.

### This is also the first numerical evidence for H18

FP32 halves peak GEMM throughput on GP100 (18.7 -> 9.3 TFLOPS). If a fraction
`f` of prefill time were GEMM-rate-bound, doubling GEMM time gives a speed ratio
of `1/(1+f)`:

- pp2048: `193.008/222.724 = 0.8666` -> **f = 15.4%**
- pp16384: `182.762/208.443 = 0.8768` -> **f = 14.1%**

**Roughly 15% of prefill time is GEMM-rate-bound; the other ~85% is not.** That
is direct quantitative support for H18 — the GDN path, which has no GEMMs, is
where prefill time actually goes. H18 was parked for want of a cheap probe;
this is that probe, and it cost one build and ten minutes.

**Caveat on the arithmetic:** this assumes cuBLAS FP32 is exactly half-rate and
that nothing else changes. The FP32 path also converts quantized weights to F32
rather than F16, which doubles intermediate memory traffic, so some of the 13%
is bandwidth rather than FLOPs. Treat ~15% as an estimate with a stated model
behind it, not a measurement.

**Fix note 2026-10-14 — this inference is wrong.** Phase 0 measured every op of the
real graph on one card (`results/phase0/`): GEMMs are **~87% of prefill op time**,
the GDN scan ~6%, flash attention ~7% at 4k. So the 13% H17 cost cannot mean "15%
GEMM-bound". The broken assumption is most likely "FP32 is exactly half-rate on
this path". The real cause is unverified and does not matter for the plan.

**Why decode barely moved:** decode is memory-bandwidth-bound, not
arithmetic-bound, so the compute mode is nearly irrelevant to it. The +0.36% is
consistent with everything else this repo has measured about the decode path.

### What it costs, and what is still unknown

At the H26 serve config, -12.3% prefill takes 100k TTFT from **~7.8 min to
~9.2 min** per uncached turn.

Against that: upstream's KL divergence 0.004962 -> 0.000001 and top-token
agreement 95.00% -> 99.89%, i.e. **~1 greedy token in 20 flips**. We have **not**
verified that on this rig — Test B was not run, so the benefit side of the trade
is still upstream's number, not ours.

**Not adopted. Not rejected. Parked for a decision**, because the throughput cost
is real and the deliverable's fourth objective is quality, which this repo has
never measured. It cannot be settled on speed alone.

### One option worth recording before anyone re-derives it

The three sites are separable. `fast_fp16_hardware_available` is the one that
sets the **cuBLAS compute type for prefill GEMMs**; the other two govern the
decode mat-vec, flash-attention tile and quantized dot products. A **partial
patch** — leave `fast_fp16_hardware_available` alone, patch the other two —
would keep FP16 prefill GEMMs and move only the decode-side arithmetic to FP32.
That would recover most of the 13% while still fixing the paths closest to the
sampled logits. Whether it keeps most of the accuracy benefit is unknown and
would need one more build (free) plus a short output-comparison run.

---

## H18 — GDN layers dominate prefill

> **Status: parked by rule zero, 2026-08-27.** Still plausible, still untested,
> but it cannot change the serve command. See "Where this stands" at the end of
> the section — read that before acting on the test plan below, which was
> written when H22 still looked like a flag.

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

**Why it *used to* decide the plan:** if true, H14 and the whole family of quant/GEMM
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

### Where this stands (2026-08-27)

**Parked by rule zero. It is a diagnostic, not a lever.** Three things moved
against it, none of which refute it:

1. **H14 argued the other way.** The `-ub` sweep was deliberately framed so a
   *null* result would be positive evidence for H18 — a token-at-a-time kernel
   should barely care about batch shape. We measured **+63%**. That is
   GEMM-shaped behaviour, so H18 is no longer the leading explanation for the
   30%-of-peak figure.
2. **Its flat-curve premise broke at depth** (H13, above). The depth decay fits
   the ~16 full-attention layers' O(n²) cost better than a kernel that is linear
   in tokens. Best current read: GDN-bound short, attention-bound long.
3. **The lever it feeds is gone.** A confirmed H18 has exactly one action
   attached — write the chunked GDN path — and that is H22, which crashes under
   `-sm tensor` because the chunked ops carry no split-axis metadata. Kernel
   work, not a patch.

So neither outcome of H18 changes the serve command today, and it does not earn
its power. **Unpark when H22 becomes runnable** (someone wiring the split axes,
or an `-sm none`/`layer` run being worth it on its own merits).

If it is ever re-run, note that step 1 above must be taken at **`-ub 2048`**;
every number in the original framing came from the `-ub 512` regime H14 showed
to be a local minimum.

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

### Verdict — CONFIRMED 2026-10-14, at default flags (`h19-prompt-cache-defaults`)

| step | prompt tokens | re-prefilled | prompt_ms | vs cold |
|---|---|---|---|---|
| turn 1 (cold) | 14,162 | 14,162 | 47,698 | — |
| turn 2 | 14,316 | 27 | 1,373 | −97.3% |
| turn 3 | 14,467 | 24 | 1,430 | −97.1% |
| turn 4 | 14,614 | 151 | 2,240 | −95.5% |
| unrelated request | 292 | 292 | 1,647 | — |
| turn 5 after the interleave | 14,765 | 24 | 1,510 | −97.0% |
| turn 5, `cache_prompt=false` (control) | 14,765 | 14,765 | 50,013 | control |

**Both predictions hold, with no flag changes.** The checkpoints work under MTP
and `-sm tensor`, and the idle-slot RAM cache brings the session back after an
interleaved request. Turn-2+ TTFT is about 1.4–2.2 s against 50 s cold.

The ~1.4 s floor for ~25 new tokens is restore and launch overhead, not prefill.
It is worth knowing about, but not worth a run. Peak temperature was 65 °C.

**The serve command is unchanged. S2 (custom GDN checkpointing) is not needed for
the single-session agent loop.** The limit that remains is `--cache-ram 8192`
across many concurrent long sessions; that only matters if the harness ever runs
more than about one 100k session at a time.

**Pre-registered 2026-10-14 (run `h19-prompt-cache-defaults`).** Reading the
server source first changed the test. Same-slot reuse doesn't go through
`--cache-ram` at all, and the defaults are already `-ctxcp 32`, `--cache-ram 8192`
and idle-slot caching on. The server also checkpoints the hybrid model's state at
the last user message and near the prompt's end, whatever the `-cms 8192` spacing.
So the first question is whether the **defaults** already reuse on our build
(MTP, `-sm tensor`). One load runs a 4-turn session on an ~11k-token system
prompt, then one unrelated request, then turn 5 of the session, then turn 5
again with `cache_prompt=false` as the cold control.
**Prediction:** on turns 2–4, prompt_n is <10% of the prompt and prompt_ms drops
>90% against the control. Turn 5, after the interleave, also comes back above
90% via the idle-slot RAM cache. If both hold, no flag goes into the serve
command. If turn 2+ re-prefills in full, the checkpoint path is broken under
MTP or tensor split, which is S2/fork work.

**Original test:** `--cache-ram 20000 --cache-idle-slots` on the WEB_BENCH agentic
scenario, measuring per-turn TTFT across a multi-turn session against the same
session with `--cache-ram 0`. This is the **highest-value practical change in the
list** for the stated use case, and it costs nothing but a flag.

---

## H20 — Qwen3.8-27B at IQ3_S on a single P100 — **CLOSED, not feasible**

### Verdict — measured and closed 2026-08-27 (user call)

**A single P100 is not feasible for real-world use.** H25 measured it at <=16k:
**56%** of the pair's prefill, **53%** of its drafter-free decode, and **MTP
does not fit at all** (OOM at 16,029 of 16,269 MiB, already with q8_0 KV) — so
against the config we actually serve with, the fallback runs at **38%**. The
VRAM arithmetic below independently caps it below 100k. Two independent
failures against the objective: it cannot reach the context target, and it
cannot use the decode lever.

**Do not spend further runs on single-GPU serving.** The remaining sub-question
(does MTP fit at a smaller `-ub`?) was explicitly declined: even its best case
leaves the fallback behind the two-card drafter-free number while also
surrendering prefill.

What survives from this line of work is not the fallback but H25's KV-quant
result, which applies to the **two-card** config: `q8_0` costs 1.5% of decode,
not the 14.3% the repo had been quoting from TurboQuant.

**Original framing, kept for the record:**

**Reframed 2026-08-26 (user call).** Originally a Qwen3.5-9B fallback —
withdrawn. A different model, on an older architecture, is not a fallback for
"serve the 27B"; it is a different deliverable, and its numbers do not transfer.
**The single-GPU arm of this project is now Qwen3.8-27B-UD-IQ3_S**, and any
single-GPU comparison in this repo is against that file. The 9B is retired from
the plan; it survives only as July NIAH history in H13's decay reasoning.

**Confirmed on disk:** `/root/Qwen3.8-27B-UD-IQ3_S.gguf`, 11,483 MiB (11.2 GiB).
No quantising needed — this is a re-run, not a build.

**Claim:** Qwen3.8-27B-IQ3_S on one P100 serves at a usable depth with decode
within 25% of the two-card `UD-Q4_K_M` config, at NIAH quality acceptable for
coding work.

**Why the claim changed shape when the model did.** The old 9B claim was a
**prefill** claim (≥2× TTFT), and it was plausible for a 9B because a smaller
model cuts the FLOP count roughly 3×. **IQ3 does not.** It is the same 27e9
parameters and therefore the same prefill arithmetic — the weights are
dequantised to f16 before the GEMM either way. IQ3 buys **bytes, not FLOPs**:

| phase | bound by | what IQ3-on-one-card does |
|---|---|---|
| prefill | compute | **worse.** Same FLOPs, but one card is 18.7 TFLOPS against two cards' 37.4. Expect roughly half the prefill unless tensor-split overhead is far worse than measured |
| decode | memory bandwidth | **plausibly competitive.** ~11.2 GiB of weights against Q4_K_M's ~16 GiB spread over two cards, and no cross-GPU traffic per token at all |

So **do not carry the old ≥2× prefill claim over to IQ3** — it was a property of
the 9B, not of the quant. This hypothesis is now about whether a single card is
*viable*, and about decode, not about winning on TTFT.

**VRAM math, and it caps the depth this fallback can reach.** One P100 is
16,269 MiB. Weights alone leave **4,786 MiB** for KV + activations + overhead:

| context | f16 KV size | fits in 4,786 MiB? |
|---|---|---|
| 64k | 4,096 MiB | barely — ~690 MiB left for activations, likely too tight at `-ub 2048` |
| 100k | 6,250 MiB | **no** — 1,464 MiB over budget at f16 KV |

As specced this fallback **cannot reach 100k on one card at f16 KV**: it tops
out below that, or needs a smaller `-ub`, or needs KV quantisation. The last
option reopens the TurboQuant question this repo priced at 14.3% of decode
(H4) — and here it may not be a cost/benefit choice but a requirement, which is
a materially different decision than the one H4 closed.

**Test:** `CUDA_VISIBLE_DEVICES=0`, `-sm none`, IQ3_S. Three steps in order:
1. **Find the achievable depth at f16 KV.** Start at 32k and step up watching
   VRAM. Do not assume 64k fits until it is measured.
2. **Prefill and decode at that depth**, against the two-card Q4_K_M numbers at
   the same depth — same prompt, same `N_PREDICT`, same reps.
3. **NIAH quality gate at that depth**, `/root/niah_test/run_engine.py`,
   fixtures already generated.

**Watch:** quality is the deeper question. IQ3 is a much more aggressive quant
than anything tested on this rig, and the NIAH harness's only baseline is the
July 9B run — so it validates the *harness*, not this model at this quant. The
Q4_K_M 27B has no NIAH baseline either, which means **step 3 needs a two-card
Q4_K_M control run or it measures nothing**. Use the multi-needle fixtures
(`niah_*_multi.jsonl`) and WEB_BENCH if it looks promising.

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

**Sized 2026-08-27.** H17's A/B put a number on it without a dedicated run: see
"This is also the first numerical evidence for H18" in H17 above. ~15% of prefill
time is GEMM-rate-bound, so ~85% is somewhere else, and the GDN path is the
obvious candidate. That is support, not proof — it bounds the GEMM share rather
than measuring the GDN share directly.

**Ordering — superseded 2026-08-27.** The original advice was "run H18 first;
H18 sizes the prize, H22 collects it." That assumed H22 was a flag away. It is
not: the crash above makes it a kernel project, so there is no prize to size and
**both are parked**. If someone annotates the split axes (or decides an `-sm
none`/`layer` run is worth it on its own), the original ordering applies again —
H18 first, at `-ub 2048`.

---

## H24 — `-ub 2048` costs decode

**Claim:** raising `-ub` from 512 to 2048 changes decode throughput at 16k by
less than 5%, so H14's prefill win is free.

**Why this has to be tested before any serve command is written.** H14 measured
`-ub 2048` as +63% on prefill and the README already carries it in the "best
measured configuration" block — but every decode number in this repo, including
the headline **29.26 t/s at 16k**, was taken at the old `-ub 512`. A serve
command sets one ubatch for both phases. If 2048 costs decode materially, the
command is a trade, not a free win, and the objective table's second and third
rows cannot both be satisfied by the same flag.

**Reasoning for the null:** decode processes one token (plus MTP draft tokens)
per step, so `n_ubatch` should not enter the decode graph at all. The plausible
mechanisms for a non-null are indirect: a larger activation arena changes
allocator behaviour or VRAM pressure at depth, or the drafter's verify step
batches differently.

**Second thing this run settles:** the 29.26 t/s figure was measured at
`GGML_CUDA_ALLREDUCE=internal`, which the repo has since replaced with `none`
(H5/H10). The control arm re-measures it under the current env rather than
quoting across a config change.

**Test:** `scripts/run-h24-ubatch-decode.sh` — two arms, `-ub 512` and
`-ub 2048`, identical otherwise (`-sm tensor`, `-b 2048`, MTP-Q4_0, `CTX=16384`,
`prompts/h11-depth-prompt.txt`, `N_PREDICT=400`, r=3, `ALLREDUCE=none`). Both
arms reload the server, because `-ub` is a launch parameter.

**Watch:** decode t/s (the claim), and **acceptance rate** as a control — ubatch
must not move it. If acceptance shifts, something other than ubatch differed
between the arms and neither number is trustworthy. Also watch VRAM: the 512 arm
loaded at 10235 MiB/card, and the extra activation arena at 2048 comes out of
what a 100k KV cache will later need.

### Result — REFUTED, but the cause is the drafter, not the decode path (2026-08-27)

`scripts/run-h24-ubatch-decode.sh`, `mainline-rebased`, `-sm tensor`,
`ALLREDUCE=none`, MTP-Q4_0, 16k prompt, 400 tokens, r=3, peak 68 °C. Raw:
`results/h11-placement.csv` (labels `h24-ub512-*`, `h24-ub2048-*`).

| | `-ub 512` | `-ub 2048` | delta |
|---|---|---|---|
| decode t/s | 29.39 | 27.41 | **−6.7%** |
| prefill t/s | 198.5 | 310.5 | **+56.4%** |
| acceptance | 73.3% | 66.4% | **−6.9 pp** |
| draft_n / accepted | 374 / 274 | 399 / 265 | |
| VRAM per card | 10,235 MiB | 10,973 MiB | **+738 MiB** |

Spread inside each arm is under 0.06 t/s on decode and 0.7 t/s on prefill, so
the deltas are far outside noise. The claim was <5%; 6.7% refutes it.

**The control variable moved, and following it explains the whole result.**
Acceptance was specified as the control precisely because ubatch should not
touch it — it did. Working it through:

- Every output token is either an accepted draft or a target forward pass, so
  400 tokens cost `400 − accepted` target steps: **126** at `-ub 512`, **135**
  at `-ub 2048`, i.e. **+7.1% more work**.
- That alone predicts a **−6.7%** decode change. Measured: **−6.7%.**

**So there is no intrinsic ubatch penalty on the decode path** — as the null
reasoning said, `n_ubatch` does not enter the decode graph. The entire loss is
the drafter proposing worse. Two consequences: a **drafter-free** config should
take `-ub 2048` for free, and the right fix is to tune the drafter at this
ubatch rather than to give back 56% of prefill.

**Quality is unaffected — verified, not assumed.** The 400 generated tokens are
**byte-identical** between the two arms, and identical across all three reps in
each arm. The target model corrects the worse drafts to exactly the same output,
which is what a correct speculative implementation must do. What changed is only
the drafter's own numerics (its prefill is chunked differently at a different
ubatch), and therefore only its efficiency.

**The trade is lopsided in favour of `-ub 2048` for any realistic turn.** At 16k
a prefill costs 81 s at 512 and 52 s at 2048 — **29 s saved per turn** — while
decode costs 2.45 ms/token more. Break-even is at **~11,800 output tokens in a
single turn**; coding turns are one to two orders of magnitude short of that.
`-ub 2048` stays in the serve command.

**Second thing this settles:** the standing 29.26 t/s headline reproduced at
**29.39 t/s** under `ALLREDUCE=none` with identical VRAM (10,235 MiB/card) and
identical acceptance (73.3%). The internal→none change did not cost decode, and
the figure can now be quoted against the current env.

**Not tested:** whether the acceptance drop persists at 64k–100k (it may grow or
vanish — H1 showed acceptance is strongly depth-dependent), and whether the
+738 MiB activation arena is affordable once a 100k KV cache is resident.

---

## H25 — the IQ3_S single-card fallback, measured at short context

**Claim (user-directed, 2026-08-27):** how does Qwen3.8-27B-UD-IQ3_S on **one**
P100 compare to the two-card pair on speed, at <=16k, across the levers already
in use — ubatch, KV cache type, and MTP?

**Test:** three single-GPU arms (`-dev CUDA0`, so `-sm` is a no-op and was not
swept; the two-card numbers already exist). `scripts/run-h25-iq3-single.sh`.
The `-ub 512` cells were deliberately cut — H14 and the f16 arm both settle that
2048 wins on prefill, and H24 settles that ubatch does not touch decode.

### Result — the fallback is roughly half the pair, and it cannot use MTP (2026-08-27)

**Single card vs the pair, at 16k:**

| | 1x P100, IQ3_S | 2x P100, Q4_K_M | ratio |
|---|---|---|---|
| prefill | 184.1 t/s | 327.1 t/s (H13) | **56%** |
| decode, no drafter | 10.4 t/s | 19.75 t/s (Phase 2) | **53%** |
| decode, best available | **10.4 t/s** (MTP does not fit) | 27.41 t/s (H24) | **38%** |

The prefill ratio is what the arithmetic predicted: IQ3 cuts weight **bytes**,
not FLOPs, and prefill is compute-bound, so it scales with card count. Decode
lands in the same place rather than better — halving the weight bytes did not
buy back the lost card.

**MTP does not fit on one card. This is the finding that decides the fallback.**
The drafter loaded (16,029 MiB of 16,269) and then died on the first request:

```
common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 99, abort
CUDA error: out of memory ... in ggml_cuda_pool_vmm::alloc
```

240 MiB free is not enough for the `-ub 2048` compute buffer — and this was
**already with q8_0 KV**, so the usual mitigation is spent. The fallback
therefore loses the single largest decode lever on this rig (+48% at 16k), which
is why its realistic gap to the working config is 38%, not 53%.

**KV quantisation is very nearly free here — and this matters well beyond the
fallback.** One card, `-ub 2048`:

| KV | pp4096 | pp16384 | tg128 |
|---|---|---|---|
| f16 | 200.96 | 183.80 | 11.38 |
| q8_0 | 202.44 | **184.13** | 11.21 |
| q4_0 | 202.05 | 184.05 | 11.18 |

**Prefill is unchanged (inside 0.2%) and decode costs 1.5%**, for either quant.
q4_0 buys nothing over q8_0 on speed, so q8_0 is the pick.

This contradicts the prior this repo has been carrying. H4 priced KV
quantisation at **14.3% of decode** — but that was measured on **TurboQuant**
specifically, and it has been quoted since as if it applied to KV quantisation
generally. Stock `q8_0` costs 1.5%. **The 14.3% figure is a TurboQuant number,
not a KV-quant number**, and should not be used to argue against `-ctk q8_0`.

**Consequence for the two-card 100k target, which is the actual deliverable:**
q8_0 KV should halve the 100k KV cache from 6,250 MiB to ~3,125 MiB for ~1.5%
of decode. That is the cheapest headroom on the table and it was being refused
on the strength of a number measured on a different technique.

**Ubatch confirms H24 independently:** drafter-free decode is 11.41 t/s at
`-ub 512` and 11.38 at `-ub 2048`. A 4x ubatch change moves decode by 0.3%,
which is the null H24 predicted once the drafter is removed.

**Closed 2026-08-27 (user call). Single-GPU is not feasible for real-world
use.** The proposed follow-up — MTP at a smaller `-ub` — was declined and is
**not parked, it is closed**: the baseline is established and the verdict does
not turn on it. Even the optimistic outcome (~15 t/s) leaves the fallback below
the two-card drafter-free number while giving up prefill as well, and the
pessimistic outcome is another OOM. Do not spend runs on single-GPU serving.


---

## H26 — does the config actually serve at 100k?

**Claim:** `-c 100000 -ub 2048 -sm tensor -ctk q8_0 -ctv q8_0` with the MTP
drafter loads and serves on two P100s, and decode at 100k stays within 25% of
the 16k figure (27.41 t/s).

**Why this outranks everything else open.** The objective is 100k. Every
serve-side number in this repo is 16k. H13's `pp100000 = 215.4 t/s` came from
`llama-bench`: no drafter, no server, no decode, no VRAM figure, no acceptance,
no thermal profile. **We have never established that the deliverable config
loads at 100k at all.** If it does not, the quant sweep and the quality gate are
both aimed at the wrong config, and every hour spent on them is wasted.

**Five open gaps close on one request,** which is why this is worth a long run
under the electricity rule:

1. Does it load? (VRAM at 100k, never measured)
2. Decode at 100k (never measured above 16k)
3. MTP acceptance at 100k (it moved 40.1% -> 73.3% from 4k to 16k; unknown at 100k)
4. `q8_0` KV on **two** cards (H25 measured it on one)
5. Sustained thermals — 100k prefill is ~8 min of continuous compute against an
   83 C limit, and nothing here has ever run hot for more than ~3 min

**VRAM arithmetic, done before spending the power.** H24 at 16k with MTP and f16
KV used 10,973 MiB/card. The repo's figure for a 100k f16 KV cache is ~6,250 MiB
total, so 16k f16 is ~512 MiB/card and the non-KV base is ~10,461 MiB/card. At
100k with `q8_0` that base carries ~1,563 MiB/card of cache, landing near
**12.0-12.5 GiB of the 15.9 GiB available**. It should fit, with the compute
buffers as the main uncertainty. `-np 1` is passed explicitly: the server
defaults to `n_slots = 4`, and a four-slot allocation at this depth is a
needless risk.

**Arms are a ladder, not a sweep** (`scripts/run-h26-100k-serve.sh`). Each runs
only if the one above it failed:

| Arm | KV | Drafter | Runs when |
|---|---|---|---|
| A | `q8_0` | MTP | always — this is the config we want |
| B | `q4_0` | MTP | only if A cannot fit. H25 showed q4_0 is no faster than q8_0, so this is purely a VRAM fallback |
| C | `q8_0` | none | only if both drafter arms fail — establishes whether 100k is reachable at all |

**Prompt:** `prompts/h26-100k-prompt.txt`, 97,620 tokens of real llama.cpp
source plus this repo's docs, ending in a question that requires reading the
code. Real corpus rather than repeated filler, because repetition would flatter
the drafter's acceptance rate and this rig is meant for coding at depth.

**Known limitation:** `REPS=2`, and rep 2's *prefill* number will be
contaminated by the server's slot-level LCP prefix reuse — the same trap that
invalidated the H11 prefill column. Rep 1 is the honest prefill measurement.
Rep 2 is kept anyway because decode is unaffected by prefix reuse, and because
a second 8-minute prefill doubles the sustained-thermal exposure, which is
itself one of the five things being measured.

_This limitation did not materialise — see the result below._

### Result — 2026-08-27

Arm A only. It fit, so arms B and C never ran.

| Metric | 100k (H26) | 16k (H24) | Ratio |
|---|---|---|---|
| Decode | **20.02 t/s** | 27.41 | 73.0% |
| Prefill | **207.4 t/s** | 310.5 | 66.8% |
| MTP acceptance | **60.3%** | 66.4% | −6.1 pts |
| VRAM | **12,147 MiB/card** | 10,973 | +1,174 |
| Peak temp | **70 C** | 68 | +2 |
| TTFT | **~7.8 min** | ~53 s | — |

Mean of 2 reps (20.00/20.03 decode, 207.71/207.13 prefill). Load was 190 s cold,
20 s warm. `n_slots = 1`, `n_ctx_slot = 100096`, `kv_unified = false`.

**Verdict: the config serves at 100k, and the claim's numeric threshold fails.**
Decode is **26.9%** below the 16k figure against a claimed ≤25%. That is a miss
of 1.9 points, and it is recorded as a miss rather than rounded away — but the
thing the run existed to establish is settled: **the deliverable config loads
and serves at the target depth**, with 4,122 MiB/card to spare.

**Five gaps closed on this one request:**

1. **It fits.** 12,147 MiB/card, ~4.1 GiB/card headroom. The VRAM estimate made
   before the run (12.0–12.5 GiB) was right, so the model of where the memory
   goes is sound. `q4_0` KV is not needed — arm B is withdrawn.
2. **Decode at 100k is 20.02 t/s**, the first such measurement here.
3. **MTP acceptance at 100k is 60.3%** — and it is *not* monotonic in depth:
   40.1% at 4k, 66.4% at 16k, 60.3% at 100k. It peaks somewhere in between.
4. **`q8_0` KV works on two cards**, as H25 predicted from one. The 100k cache
   cost ~1,174 MiB/card over the 16k run; at f16 it would have been ~2,350.
5. **Sustained thermals are a non-issue.** 70 C peak across two consecutive
   ~8-minute prefills — the same peak as the 3-minute 16k runs, 13 C under the
   abort limit. The 175 W cap is doing the work.

**Cross-harness agreement, which had never been checked.** `llama-server`
measured 207.4 t/s prefill at 100k against `llama-bench`'s 215.4 (H13) — 96%
agreement between two different harnesses on two different days. Neither number
is obviously wrong, and the small gap is the expected cost of the chat template
and HTTP path.

**The rep-2 prefill contamination did not happen.** 207.13 vs rep 1's 207.71:
prefill was genuinely re-done, so both reps are valid measurements. The H11
prefix-reuse trap is therefore *not* universal. Cause unconfirmed — `-np 1` with
`kv_unified = false` is the obvious suspect, but this run did not isolate it, so
do not rely on `cache_prompt: false` working until something does.

---

## H27 — CUDA graphs on sm_60 (fork D1)

**Claim.** Upstream disables CUDA graphs below Volta (`ggml-cuda.cu`, a 2024 perf
judgement). The fork's `GGML_CUDA_PASCAL_GRAPHS=1` re-enables them.

**Evidence for.** Phase 0.2 (`results/phase0/README.md`): 2-card decode is idle 23% per
token (~13.6 ms, ~10 ms excluding trace overhead) with under 1 ms of copies — launch and
sync latency across ~2.2k kernels per token. 1-card decode is idle only 4%.

**Pre-registered prediction** (fork `0cdf2be`, drafter-free, 16k depth, 400 tokens):
- 2-card Q4_K_M `-sm tensor`: decode **+8% or more**. Prefill within ±2%.
- 1-card IQ3_S `-sm none`: decode **+4% or less**. Prefill within ±2%.
- Greedy output **byte-identical** to graphs off. If it differs, the lever is rejected
  whatever the speed.
- It may also fail outright under the meta backend (cross-device copies during capture).
  That is a result, not a reason to patch blindly.

**Run.** `scripts/run-h27-graphs.sh`: four arms, off/on × 2-card/1-card, REPS=2. Also
RUNBOOK §7 item 6 (D5): `GGML_CUDA_P2P=1` with MTP on the banked build, one arm against
the `h24-ub2048-mtp-16k` control (27.41 t/s, 66.4%).

### Result (2026-10-14) — REFUTED

Two reps each, 16k depth, 400 tokens, drafter-free, fork `0cdf2be`. Rows are in `results/h11-placement.csv`.

| Arm | Decode t/s | Prefill t/s (rep 2) | VRAM/card |
|---|---|---|---|
| 2-card graphs off | 19.89 | 342.2 | 9,073 |
| 2-card graphs on | 19.66 (**−1.1%**) | 342.0 | 9,113 |
| 1-card graphs off | 11.00 | 196.6 | 13,161 |
| 1-card graphs on | 10.95 (**−0.5%**) | 197.1 | 13,177 |

- **Greedy output is byte-identical** with graphs off and on, on both configurations.

### Correction (2026-10-14) — output identity was compared on the wrong field

The original comparison read only `message.content`. Within 400 tokens the model is still
thinking, so `content` is empty in every arm and the check was vacuous. Re-compared on
`reasoning_content + content` with `scripts/compare-outputs.py` (rep 1 of each arm):

| Pair | One lever? | Result |
|---|---|---|
| H27 graphs off vs on, 2-card and 1-card | yes | **identical** — the H27 conclusion stands |
| `h24-ub2048-mtp-16k` vs `d5-p2p-mtp-16k` (P2P) | yes | **identical** |
| `p2-tensor-none-16k` vs `p2-tensor-mtp-16k` (drafter, banked, `-sm tensor`) | yes | **identical** |
| `p2-layer-none-16k` vs `p2-layer-mtp-16k` (drafter, banked, `-sm layer`) | yes | **differ at char 183** |
| `h24-ub512-mtp-16k` vs `h24-ub2048-mtp-16k` (ubatch) | yes | **differ at char 183** |
| `h28-2card-mmvh1` vs `h28-2card-mmvh1-mtp-p2p` (drafter + P2P; P2P shown inert above) | ~yes | **differ at char 369** |
| H28/H29 mmvh off vs on | yes | differ (expected; lossy, KLD-gated) |

Consequences:
- **H24's "byte-identical, the target corrects to the same output" is wrong.** `-ub` changes
  the target's own prefill chunking, so the KV cache and the greedy path differ slightly.
  That is a numerics change, not a quality regression, and `-ub 2048` stays; but part of
  the 6.9 pp acceptance drop may be the target taking a different path, not only the drafter.
- **The drafter can change greedy output.** It held on the banked `-sm tensor` build and
  failed on `-sm layer` and on the fork with mmvh. Batched verify runs the target with
  ncols>1, which can take different kernels and reduction orders than single-token decode.
  QUALITY-PLAN Sweep D's premise ("output is a property of the target") does not hold in
  general, so **Phase 8 gates the exact serve command, drafter included.**
- **Graphs really engage.** A 1-card `GGML_CUDA_GRAPH_DEBUG=1` run (`logs/h27-graph-debug.log`) completed warmup and replayed 14 of 16 tokens. No arch disable fired. So the null result is real, not a dead flag.
- **Why there's no gain:**
  - On 1 card, the GPU was already 96% busy (Phase 0.2). The CPU was issuing launches ahead of the GPU, so removing launch cost frees nothing.
  - On 2 cards, the 23% idle is the sync and AllReduce round trips between ~130 subgraphs per token. Graphs can't remove those.
- **2-card prefill rep 1 is low on the graphs-off arm** (318 vs 342) because it is the first request after load. Prefill is compared on rep 2.
- **Decision.** D1 is closed and the flag stays off. On 2 cards the decode lever is D3: a one-shot P2P reduce, which cuts round-trip latency. On 1 card it is D2: the matvec kernel.

### D5 (RUNBOOK §7 item 6) — P2P with MTP, same session

On the banked build (`dflash2-rebased`), with `GGML_CUDA_P2P=1`, MTP, `-ub 2048`, 16k:

- **Decode: 28.43 / 28.35 t/s**, against the `h24-ub2048-mtp-16k` control's 27.41. That is **+3.6%**.
- **Acceptance is identical** (265/399 = 66.4%), so the gain is in the transport, not the drafter.
- **Prefill is 307.7 on rep 2** vs 310.5, a −0.9% change. That matches H10's −0.3% within noise.
- VRAM is unchanged (10,973 MiB per card).

**Conclusion: P2P helps with MTP too. Adopted into the serve command 2026-10-14 (user call).**


## H28 — the Pascal half2 matvec (fork D2, `GGML_CUDA_PASCAL_MMVH=1`) lifts decode

**Basis.** On sm_60 the mmvq matvec is ALU-bound (~200 GB/s, emulated dp4a). The fork's
`mmvh` kernel (`4826395`: Q4_K, Q5_K, IQ4_XS, ncols 1–8) runs n=1 at model shapes
×1.16–2.01 in `test-backend-ops perf` (300–380 GB/s on the big matrices). Those three types
are 89% of the weight bytes on 2-card UD-Q4_K_M, but only 24.5% (IQ4_XS) on 1-card UD-IQ3_S.
It accumulates 16 products in fp16 per unit, so it is lossy relative to int8 dot products.

**Prediction (registered before the run).**
- **Accuracy gate, first.** `llama-perplexity` KLD on 8×512 tokens of `prompts/h26-100k-prompt.txt`,
  2-card `-sm layer`, `-ub 8` so the matvec path runs. The base is mmvq with mmvh off.
  The mmvh arm's mean KLD is **≤ 2× the reference**, where the reference is the same model through
  mmq (`-ub 512`, mmvh off), the numeric noise stock llama.cpp already accepts. Same-top-token
  agreement is within 0.5 pp of the reference. If the gate fails, the speed arms are skipped.
- **2-card, drafter-free, 16k:** decode **+12% or more** against `h27-2card-graphs0` (19.89 t/s,
  same fork code with mmvh off). Prefill is unchanged (±2%; prefill uses mmq).
- **1-card IQ3_S, drafter-free, 16k:** decode **+3–8%** against `h27-1card-graphs0` (11.00).
  This is small because only IQ4_XS is covered so far.
- **2-card serve config** (fork, MTP, P2P, mmvh on): decode beats the serve row's 28.39 t/s
  (`d5-p2p-mtp-16k`, banked binary) by **8% or more**. This arm changes the binary as well as the
  kernel; it answers whether the serve command should move to the fork, not what mmvh alone is worth.
- Greedy output is **not** expected to be byte-identical; the KLD gate is the quality check.

**Run.** `scripts/run-h28-mmvh.sh`.

### Result (2026-10-14) — CONFIRMED (speed), gate passed with a tail caveat (quality)

Fork `4826395`, 16k depth, 400 tokens, 2 reps. Rows are in `results/h11-placement.csv` and `results/h28-kld.csv`.

| Arm | Control | Decode t/s | Δ | Prefill (rep 2) |
|---|---|---|---|---|
| 2-card drafter-free | `h27-2card-graphs0` 19.89 | **25.29** | **+27%** (pred ≥ +12%) | 342.3 vs 342.2 |
| 1-card IQ3_S drafter-free | `h27-1card-graphs0` 11.00 | **11.82** | **+7.5%** (pred +3–8%) | 196.4 vs 196.6 |
| 2-card serve config (fork + MTP + P2P) | `d5-p2p-mtp-16k` 28.39 | **37.66** | **+32.6%** (pred ≥ +8%) | 320.3 vs 307.7 |

- MTP acceptance is 68.2% (268/393) against 66.4%. VRAM is unchanged. Peak temperature was 65 °C.
- **KLD gate PASS** (8×512 tokens, base = mmvq at `-ub 8`):

| | mean KLD | median | 99% | 99.9% | max | same top-1 |
|---|---|---|---|---|---|---|
| ref: stock mmq | 0.0166 | 0.00053 | 0.077 | 1.16 | 14.3 | 97.35% |
| mmvh | 0.0217 | 0.00004 | 0.020 | **4.30** | **24.6** | 99.17% |

- On 99% of tokens mmvh is closer to the int8 matvec than stock mmq is. **The 0.1% tail is about 4× worse.** ln(PPL ratio) is +0.5% ± 0.65%, which is not significant.
- **The suspected cause, IQ4_XS's +128 bias cancellation, is refuted** (`results/h28-kld-split.csv`, a temporary per-type switch):
  - Without IQ4_XS the tail is unchanged (99.9% 4.30, max 24.5).
  - IQ4_XS alone gives 99.9% 0.51, max 5.7.
  - So the tail comes from the Q4_K/Q5_K paths, or from 1–2 knife-edge tokens: 99.9% of 2048 tokens is ~2 tokens, and stock mmq also reaches a max of 14.3.
  - At n=2048 the mean difference (0.0217 vs 0.0166, ±0.013) is not significant.
  - More KLD tokens would not decide adoption; the Phase 8 task-level gate does.
- **Not adopted yet.** Two open items before the serve command can move to the fork binary:
  - Isolate and fix the KLD tail.
  - Run the Phase 8 quality gate.

## H29 — mmvh with IQ3_S and IQ3_XXS lifts 1-card decode further

**Basis.** The fork's `428fecb` adds IQ3_S and IQ3_XXS to mmvh, with the grids in shared
memory. At model shapes they run n=1 at ×1.7–1.9 (~210 GB/s against ~115). Covered types are
now 77% of the 1-card UD-IQ3_S weight bytes: IQ3_S 35%, IQ4_XS 24.5%, IQ3_XXS 17.6%.

**Prediction (registered before the run).**
- **KLD gate:** same method as H28, on the 1-card IQ3_S model (`-sm none`). Mean KLD ≤ 2× the
  stock mmq reference, and same-top-1 within 0.5 pp of it. If the gate fails, the speed arm is skipped.
- **1-card IQ3_S drafter-free, 16k:** decode **+15–25%** against `h27-1card-graphs0` (11.00 t/s).
  H28's IQ4_XS-only arm reached 11.82. Prefill is unchanged.
- The 2-card arm is not re-run: UD-Q4_K_M has no IQ3 tensors, so the H28 result stands.

**Run.** `scripts/run-h29-mmvh-iq3.sh`.

### Result (2026-10-14) — CONFIRMED, above the predicted range

Fork `428fecb`, 1-card UD-IQ3_S, `-sm none`, 16k depth, 400 tokens, 2 reps.

- **Decode: 15.87 / 15.86 t/s** against `h27-1card-graphs0` at 11.00. That is **+44%** (predicted +15–25%).
  H28's IQ4_XS-only arm reached 11.82, so the IQ3 types account for most of the gain.
- Prefill is 196.4 against 196.6, unchanged. VRAM is unchanged. Peak temperature was 65 °C.
- **KLD gate PASS** (`results/h29-kld.csv`):

| | mean KLD | median | 99.9% | max | same top-1 |
|---|---|---|---|---|---|
| ref: stock mmq | 0.0076 | 0.00057 | 0.151 | 4.1 | 97.65% |
| mmvh | 0.0091 | 0.00005 | 0.164 | 16.8 | 99.31% |

- Unlike H28, the 99.9% tail is level with the reference. The max is again a single token.
- 1 card is now at 15.9 t/s decode drafter-free. Same status as 2-card: not in any serve command until Phase 8.


## H30 — Phase 8 tier 1: where the quality loss comes from (KLD ladder)

**Why now.** Phase 8 needs a referent (QUALITY-PLAN §2.1) and H17 was parked "for a decision
that needs quality data". Before spending ~8 min per 100k prefill on NIAH and code
questions, one cheap ladder says which levers are big enough to be worth a long-context arm.

**Reference.** `UD-Q5_K_M` (the largest quant on disk) on the fork built with
`-DGGML_CUDA_PASCAL_SM60_FP32=ON` (`build-h17`, the H17 fix as a build option, fork 50b952f).
That is the most exact arithmetic and the finest quant this rig has.
Everything is 2-card `-sm tensor`, `ALLREDUCE=none`, `P2P=1` (the serve split), 8×512
tokens of `prompts/h26-100k-prompt.txt`, as in H28/H29. `-ub 512` = prefill numerics,
`-ub 8` = decode-path numerics (matvec).

| Arm | Differs from ref by | Measures |
|---|---|---|
| `q5-fp16` | FP16 fast path | H17 alone |
| `q4-fp32` | Q4_K_M | quant alone |
| `q4-fp16` | quant + FP16 | today's serve prefill |
| `q4-fp16-ub8` | + matvec path | today's serve decode |
| `q4-fp16-mmvh` | + mmvh | candidate serve decode |
| `iq3-1card-ub8` / `-mmvh` | IQ3_S, 1 card | 1-card current / candidate |

**Predictions.**
- H17 alone: mean KLD **0.002–0.01**, well under the quant step (`q4-fp32`, ~0.01–0.03).
  Upstream measured median 0.005 on another model.
- mmvh adds **≤ 0.005** mean KLD over `q4-fp16-ub8`, consistent with H28's +0.005.
- `iq3` arms are 2–4× the Q4 arms.

**Decision rules.**
- If H17 alone is **≥ 1/3 of the quant step**, H17 gets a tier-2 (100k) arm and goes to the
  user with the number against its −13% prefill. Otherwise H17 is closed as negligible.
- If `q4-fp16-mmvh` is within **+25%** of `q4-fp16-ub8` mean KLD, mmvh numerics are no
  bigger than noise next to the quant step, and tier 2 runs the candidate serve command
  against the Q5 reference. The current serve command gets a tier-2 arm only if the
  candidate fails there.
- Tier 2 (100k NIAH + code questions) gates the **exact serve command, drafter included**,
  because the H27 correction showed the drafter can change greedy output.

Driver: `scripts/run-h30-kld-ladder.sh`. ~8 short runs, ~25 min of GPU.

**Result (2026-10-14, `results/h30-kld.csv`).** All arms rc=0, peak 65 °C.

| Arm | Mean KLD | p99 | Same top |
|---|---|---|---|
| `q5-fp16` (H17 alone) | 0.0182 | 0.051 | 97.50% |
| `q4-fp32` (quant alone) | 0.0592 | 0.375 | 96.62% |
| `q4-fp16` (serve prefill) | 0.0624 | 0.318 | 95.83% |
| `q4-fp16-ub8` (serve decode, mmvq) | 0.0560 | 0.350 | 96.23% |
| `q4-fp16-mmvh` (candidate decode) | 0.0568 | 0.372 | 96.32% |
| `iq3-1card-ub8` | 0.1572 | 2.58 | 92.99% |
| `iq3-1card-mmvh` | 0.1638 | 2.92 | 92.75% |

- **Predictions: the scale was wrong, the ordering held.** H17 alone is 0.018 (predicted
  0.002–0.01) and the quant step is 0.059 (predicted 0.01–0.03). IQ3 is 2.8× Q4 (predicted 2–4×).
- **H17: closed as negligible**, by the pre-registered rule, though it is a near miss: alone it is
  0.31× the quant step, just under the 1/3 line. What decides it is that the errors don't
  stack. On Q4, FP16 arithmetic adds only +0.003 (0.0592 → 0.0624, +5%), because the quant
  error dominates. Buying that back would cost the −13% prefill. Revisit only if the serve quant
  moves to Q5 or above.
- **mmvh: within noise.** +1.4% over mmvq on Q4 and +4.2% on IQ3, against a +25% allowance.
  So tier 2 runs the candidate serve command against the Q5 reference.
- Side finding: the decode path (ub 8) sits *closer* to the reference than the prefill
  path (ub 512, FP16 cuBLAS): 0.056 vs 0.062.
- The p99.9 of 13–21 on every Q4/IQ3 arm (vs 0.45 for Q5) is a handful of tokens where Q4
  flips a confident prediction. It is a property of the quant, not of any kernel lever.


## H31 — F16 AllReduce payload for 2-card prefill (fork P1)

**Basis.** Phase 0.2: 2-card prefill is idle 18% of the time (0.95 s per 2048 ubatch,
10.6 GB copied per card), fully serialised with compute. Under `ALLREDUCE=none` the meta
backend's butterfly copies each F32 partial to the peer and adds it.

**Change** (fork `ggml-backend-meta.cpp`, env `GGML_PASCAL_F16_REDUCE=1`): partials of
≥ 1 MiB (prefill; decode and spec-verify stay F32) are converted to F16 on the sender,
copied, and added in F32 on arrival. Each card also rounds its *own* partial to F16
first, so both cards hold bit-identical sums (an asymmetric version would let the
"replicated" residual stream drift apart between cards).

**Prediction** (16k depth, drafter-free, against banked `h28-2card-mmvh1`, prefill 342.25 rep 2):
- Prefill **+6–10%**: about half the 18% comms share, minus ~1% for the three extra
  elementwise passes per reduce.
- Decode **±1%** (unchanged path).
- KLD vs the same build with the env off, `-ub 512`: mean **< 0.002**, same-top **≥ 99%**.
  Gate: fail either and the lever is rejected regardless of speed.
- 1-card arm: **not run**. The change only exists in the multi-device reduce; one card never
  reaches it, so a 1-card arm would re-measure a known control (rule zero).

**Driver.** `scripts/run-h31-f16-reduce.sh`: KLD pair, then one REPS=2 speed arm.


## H33 — cuBLAS algorithm choice at the prefill GEMM shapes (fork P3 probe)

**Basis.** Phase 0.1/0.2: prefill GEMM is 64% of 2-card time and runs at 12–13 TFLOPS
(~67% of FP16 peak). ggml calls `cublasGemmEx(OP_T, OP_N, F16, COMPUTE_16F,
CUBLAS_GEMM_DEFAULT_TENSOR_OP)`; on sm_60 that is cuBLAS's heuristic pick.

**Probe** (`scripts/gemm-algo-sweep.cu`, one card, no model, seconds of GPU): the same
call at N=2048 for the per-card `-sm tensor` shapes and the full 1-card shapes, every
`CUBLAS_GEMM_ALGO0..23` plus default, COMPUTE_16F and COMPUTE_32F.

**Prediction.** Default is within 3% of the best algorithm at every shape (cuBLAS's
heuristics are tuned for big square-ish HGEMMs like these). Decision: if some algorithm
wins by ≥ 5% at shapes covering ≥ half the GEMM FLOPs, add a per-shape algorithm pick to
the fork and A/B it end to end; otherwise P3 moves to the dequant fusion sub-target.
No 1-card/2-card serve arm: this is a kernel-level probe (the 1-card shapes are in it).

**Result (2026-10-14, `results/h33/`).** Prediction wrong. Interleaved confirm (6 rounds,
algorithms alternating so power-cap clock drift hits all equally), speed-up of ALGO3 over
default at n=2048:

| Shape (M×K) | where | ALGO3 / default |
|---|---|---|
| 8704×5120 | 2-card FFN gate/up | **1.108** |
| 5120×8704 | 2-card FFN down | 1.034 |
| 6144×5120, 5120×6144, 3072×5120, 5120×3072 | 2-card GDN/attn proj | 1.005–1.027 |
| 5120×17408 | 1-card FFN down | **1.250** |
| 12288×5120, 10240×5120 | 1-card proj | 1.16, 1.12 |
| 17408×5120 | 1-card FFN gate/up | 1.002 |

No shape loses at n=2048. The real surprise is smaller n (`small-n.csv`): at **n=512 the
default pick collapses to 5–7 TFLOPS** on 7 of 10 shapes while ALGO3 holds 12–15
(1.7–2.7×), and n=1024 is similar. That alone explains much of the old `-ub 512` vs
`-ub 2048` gap (H24). Below n=256 ALGO3 is 2–3× *slower*. Strictly the decision rule
(≥ 5% on ≥ half the GEMM FLOPs) is narrowly missed at n=2048 for 2-card (FFN gate/up is
~41% of FLOPs), but no shape loses, so no per-shape table is needed: the fork change is a
single condition, `ALGO3 when n >= 512` on sm_60 behind `GGML_CUDA_PASCAL_GEMM_ALGO=1`.
FLOP-weighted estimate at ub 2048: ~5% of GEMM time, so **~+3% 2-card prefill, ~+8% 1-card**.
Every prompt's last partial ubatch (n in 512–2047) gains more.

## H34 — fork levers on one rebuilt binary (Q6_K mmvh, GEMM ALGO3, F16 reduce)

**Changes** (all env-gated, default off):
- `GGML_CUDA_PASCAL_MMVH=1` now also covers **Q6_K** (1.7 GB of Q4_K_M, ~11% of decode bytes,
  including the 1 GB output head). Decode only (n ≤ 8).
- `GGML_CUDA_PASCAL_GEMM_ALGO=1`: H33's ALGO3 for n ≥ 512. Prefill only.
- `GGML_PASCAL_F16_REDUCE=1`: H31 unchanged.

**Design.** Q6_K and ALGO3 touch disjoint paths (n ≤ 8 vs n ≥ 512), so one arm measures
both: its decode vs the banked control is the Q6_K effect, its prefill is the ALGO3 effect.
H31's speed arm then adds F16_REDUCE on top (one lever) and is compared with that arm, not
with the banked control, so Q6_K can't leak into H31's decode number.

**Gates.** test-backend-ops MUL_MAT q6_K (CUDA vs CPU) passes. KLD: Q6_K at the banked H28
config (`-sm layer`, ub 8) mean ≤ 0.026 (H28 mmvh was 0.0217) and same-top ≥ 98.9%;
ALGO3 and F16 reduce each vs their own all-off base at ub 512, `-sm tensor`: mean ≤ 0.002,
same-top ≥ 99%. If the Q6_K gate fails, no speed arm runs (they all use MMVH=1).

**Predictions** (16k, drafter-free; controls h28-2card-mmvh1 25.29 / 342.25,
h29-1card-mmvh-iq3 15.86 / 196.40):
- `h34-2card`: decode **+3–6%** (Q6_K was on mmvq at ~40% of bandwidth), prefill **+2–4%**.
- `h34-1card-iq3`: decode **+0–1%** (IQ3_S has only 0.33 GB of Q6_K; output head is Q5_K,
  already mmvh), prefill **+5–10%**.
- `h34-2card-f16reduce` vs `h34-2card`: H31's prediction, prefill +6–10%, decode ±1%.

**Driver.** `scripts/run-h34-fork-levers.sh` (supersedes `run-h31-f16-reduce.sh`).

**Result (2026-10-14, `results/h34-kld.csv`, `results/h11-placement.csv`, commit 13d2cb5).**

Correctness: all 38 `test-backend-ops MUL_MAT q6_K` cases at n ≤ 8 (the mmvh path) pass. The
binary then aborts at `ggml-cuda.cu:1506` (`GGML_ASSERT(convert_func != nullptr)`) on a
non-contiguous Q6_K cuBLAS case. That abort predates the change (same with `MMVH=0`) and is
outside mmvh. The driver's `FAIL` grep does not catch a crash, so this was checked by hand.

| KLD gate | mean | same-top % | gate | |
|---|---|---|---|---|
| `q6k-mmvh-ub8` | 0.021324 | 99.216 | ≤ 0.026, ≥ 98.9 | pass |
| `gemm-algo-ub512` | 0.003662 | 97.549 | ≤ 0.002, ≥ 99 | **fail** |
| `f16-reduce-ub512` | −0.000005 | 100.000 | ≤ 0.002, ≥ 99 | pass (exact) |

ALGO3 failed, so every speed arm ran with `GEMM_ALGO=0` (the driver's pre-registered rule).

| Arm (rep 2) | decode t/s | prefill t/s | vs control | predicted |
|---|---|---|---|---|
| `h34-2card` (MMVH Q6_K) | 26.62 | 337.16 | decode **+5.2%**, prefill −1.5% | decode +3–6% ✓ |
| `h34-1card-iq3` | 15.83 | 196.72 | flat / flat | decode +0–1% ✓ |
| `h34-2card-f16reduce` | 26.64 | 360.71 | prefill **+6.8%** vs `h34-2card`, decode flat | +6–10%, ±1% ✓ |

Prefill predictions for `h34-2card` (+2–4%) and `h34-1card-iq3` (+5–10%) were ALGO3's and are
untested. The −1.5% on `h34-2card` prefill (both reps, vs 343.9/342.3 banked) is small but
consistent. Its cause is unknown: the new path is n ≤ 8 only, so it should be build drift. Not
chased; `h34-2card` is the control for everything that follows.

**Verdict.**
- **Q6_K mmvh: adopt** (inside `MMVH=1`). 2-card decode 25.29 → 26.62 t/s.
- **F16 reduce (H31): adopt.** It is numerically exact on this text: each card rounds its own
  partial too, so after the reduce the F16 rounding is the only change, and it is below the
  logit noise floor. Prefill 337 → 361 t/s at 16k.
- **ALGO3: not adopted yet.** The gate compared ALGO3 against the default FP16 GEMM, i.e.
  one FP16 accumulation order against another, so it measures *difference*, not *loss*. The
  97.5% top-1 agreement matches H17's FP16-vs-FP32 gap (q5-fp16 97.5%). That pattern fits
  "different rounding", not "worse". A gate that could tell those apart is H35.

## H35 — does ALGO3 add loss, or only different rounding?

**Basis.** H34 gated ALGO3 (H33) against the default FP16 GEMM, so it compared two FP16
accumulation orders and measured *difference*: mean 0.0037, same-top 97.5%, a fail.
H17's FP16-vs-FP32 gap shows the same 97.5% top-1 agreement. Two roundings of the same
sum disagree that much without either being worse. This was not caught at
pre-registration. The fair question is H30's: how far is each from exact arithmetic?

**Run.** H30's reference (Q5_K_M on `build-h17`, sm_60 FP32), regenerated because the H30
base file was deleted. One arm, `q4-fp16-algo3`, uses H30's `q4-fp16` config with ALGO3 on.
Its control is the banked `q4-fp16` (mean 0.062355, same-top 95.833), not a re-run: that
path is env-gated and unchanged in the rebuilt binary, and the reference is deterministic.

**Prediction.** ALGO3 lands within ±3% of the control: mean 0.0605–0.0642, same-top
95.3–96.4. Rounding order is noise next to the quant step.

**Gate.** Mean ≤ 0.062355 × 1.05 = **0.0655** and same-top ≥ **95.33**. Fail → ALGO3 stays
off, and P3 moves on to the dequant/convert sub-targets.

**Speed arms** (only if the gate passes; one lever, ALGO3, on top of H34's adopted set):
- `h35-2card-algo3` vs `h34-2card-f16reduce` (26.64 / 360.71): prefill **+2–4%**, decode ±1%.
- `h35-1card-iq3-algo3` vs `h34-1card-iq3` (15.83 / 196.72): prefill **+5–10%**, decode ±1%.
- 1-card KLD: **not run** (rule zero). It is the same cuBLAS algorithm on the same sm_60
  hardware with bigger M. The 2-card gate covers the numerics question, and 1-card is
  the secondary target.

**Driver.** `scripts/run-h35-algo3-loss.sh`. Two short KLD runs (~5 min), then two REPS=2
speed arms if the gate passes.

**Result (2026-10-14, `results/h35-kld.csv`, commit 00e4f76).**

| vs Q5 FP32 reference | mean | p99 | p99.9 | same-top % |
|---|---|---|---|---|
| `q4-fp16` (banked H30, default GEMM) | 0.062355 | 0.318 | 18.52 | 95.833 |
| `q4-fp16-algo3` | **0.061941** | 0.302 | 18.43 | 95.686 |

Gate passed: −0.7% mean (≤ +5% allowed), −0.15 points top-1 (≤ −0.5 allowed). Inside the
predicted ±3% band. ALGO3 rounds differently but is no farther from exact arithmetic. H34's
gate was the wrong question.

| Speed arm (rep 1 / rep 2) | decode t/s | prefill t/s | vs control | predicted |
|---|---|---|---|---|
| `h35-2card-algo3` | 26.61 / 26.62 | 373.88 / 371.83 | prefill **+3.1%** (rep 2), decode flat | +2–4% ✓ |
| `h35-1card-iq3-algo3` | 15.84 / 15.84 | 195.97 / 201.09 | prefill **+1.0%** (mean), decode flat | +5–10% ✗ |

The 1-card shortfall is not a missed path: every cuBLAS GEMM goes through the patched
`cublasGemmEx` call, and sm_60 has no dp4a, so MMQ never runs. Two likely causes:
- The 1-card FFN gate/up shape (17408×5120, ~41% of GEMM FLOPs) gains nothing (H33: 1.002).
- Sustained 1-card GEMM sits at the 175 W cap, where a more efficient kernel buys clocks, not
  throughput. H33's probe was short bursts.

Not chased: 1-card is the secondary target and the lever costs it nothing.

**Verdict: adopt `GGML_CUDA_PASCAL_GEMM_ALGO=1`.** The 2-card drafter-free prefill at 16k
goes 342.25 (H28) → 371.83 with F16 reduce + ALGO3, **+8.6%** in all. The H32 candidate
carries all three levers.

## H32 — Phase 8 tier 2: 100k haystack gate for the candidate serve command

**Arms** (QUALITY-PLAN §0; `scripts/run-phase8-tier2.sh`, 19 questions from
`prompts/phase8-fixture.json` over one ~98k-token prefix, thinking off, greedy):
- `p8-ref`: `build-h17`, Q5_K_M, no drafter, mmvh off.
- `p8-candidate`: fork, Q4_K_M, MTP + P2P, `GGML_CUDA_PASCAL_MMVH=1`, `GGML_PASCAL_F16_REDUCE=1` (H34)
  and `GGML_CUDA_PASCAL_GEMM_ALGO=1` (H35), the full adopted set.
- `p8-current` only if the candidate misses a question the reference hits.
- `p8-1card` (IQ3_S, `-sm none`, mmvh, no drafter) only if it loads at 100k.

**Predictions.**
- Reference: **≥ 17/19** (needles 5/5; function-name questions are the hard layer).
- Candidate: **the same set as the reference.** H30 puts its whole numeric loss at 0.057
  mean KLD, mostly the quant step, which rarely flips a fact retrieved at short answer length.
- 1-card IQ3_S (if it runs): **≥ 2 fewer hits** than the reference (KLD 2.8× the Q4 arms).

**Pass rule:** the candidate hits every question the reference hits, and it then replaces the serve
command. A miss triggers `p8-current` to tell whether the fork or the quant is at fault.

**Result (2026-10-14, `results/phase8-tier2.csv`, `results/phase8/*.jsonl`, commits 74812a4, cc2e345).**

| Arm | needles (L1) | function names (L2) | total | 98k prefill + 1st answer |
|---|---|---|---|---|
| `p8-ref` (Q5_K_M, FP32, no drafter) | 5/5 | 13/14 | **18/19** | 608 s |
| `p8-candidate` (fork Q4_K_M + MTP + P2P + MMVH + F16 reduce + ALGO3) | 5/5 | 14/14 | **19/19** | 440 s |

- Predictions held. Reference ≥ 17 ✓. Candidate hits the same set ✓ (a superset).
- The reference's only miss is a near-miss. At depth 87 it answered
  `llama_get_sampled_logits_ith` for `llama_get_logits_ith`; the candidate got it right. One
  question can't rank the two arms. It only shows the gate isn't saturated by the
  reference's own errors.
- **Pass: the candidate replaces the serve command.** `p8-current` isn't needed (it runs only on a
  candidate miss). `p8-1card` was skipped as pre-registered: IQ3_S doesn't hold 100k on one card.
- Peak temperature 66 °C on both arms.
- **16k confirmation** (`serve-fork-16k`, commit 1e285c1): the adopted command (f16 KV, REPS=2) does
  39.48 t/s decode, 353.0 t/s prefill, 72.3% acceptance, against the control
  `h28-2card-mmvh1-mtp-p2p` at 37.66 / 314.3 t/s. That is +4.8% decode and +12.3% prefill,
  inside the predicted +3–6% / +8–10% on decode and slightly above on prefill.

## H36 — full-duplex AllReduce copies (fork P2 step 0 finding)

**Why.** P2 step 0 (bench a16a6c2: pp2048 nsys with P1 + ALGO3 + mmvh) left 10.9% of prefill wall
idle on each card, all of it inside the reduce. Each reduce copies 0→1 (21 MB, 2.06 ms), then 1→0
(2.05 ms) starting 9 µs after the first ends, so each card idles for both copies. The cause is the
issue order: `cpy_tensor_async` makes card 1's stream wait for the 0→1 copy *before* card 1 queues
its own 1→0 copy. PCIe is full duplex, so the two copies can run at once. The fork flag defers that
wait to card 1's next graph, sync or transfer. This replaces the chunked-GEMM P2 design for now:
it targets the same idle with no GEMM split.

**Claim.** With the flag, 2-card prefill rises and the numerics are unchanged.

**Test** (`scripts/run-h36-duplex-reduce.sh`, one lever):
1. Identity gate. H35's perplexity config (Q4_K_M, ub 512, 8 chunks, ALGO3, P2P): flag off saves the logits,
   flag on is scored against them. Pass = mean KLD ≤ 1e-5 and same-top 100%.
2. Speed. The serve command at 16k (MTP, all levers, f16 KV, REPS=2) plus the flag. Control is
   `serve-fork-16k` banked today: 39.48 / 353.05 t/s.
- 1-card arm skipped (rule zero): the flag only touches peer copies, and one card makes none.

**Prediction.** Prefill +3–5%. The ceiling is ~5.4% at pp2048, a little less at 16k depth where
attention takes a bigger share. Decode within ±1%: tg copies are F32 and small (~25 ms per 64 tokens).

**Result (2026-10-14, bench de07962, fork 897d8ee).**
- Identity: flag on vs off, mean KLD −0.000005, same-top 100%. The flag only reorders work.
- 16k speed, REPS=2: 40.46 / 40.27 t/s decode and 364.2 / 366.4 t/s prefill, at 72.2–72.4% acceptance.
  Against `serve-fork-16k` (39.48 / 353.05) that is **+2.3% decode** and **+3.5% prefill**. Peak 65 °C.
- Prefill landed inside the predicted +3–5%. Decode beat the ±1% prediction, which was wrong: each
  decode reduce is latency-bound (~100 µs from matvec to add, ~124 reduces per token). Duplex
  removes one serialised copy hop from each.
- **Adopted** into the serve command. Output is bit-identical, so H32's quality pass still covers it.


## H37 — one-shot small-message AllReduce (fork plan D3)

**Why.** The tg64 decode trace (53.6 ms/token, drafter-free) is 37% idle. About 124 reduces per
token each leave a ~100 µs gap between matvec and add, roughly 12 ms/token. Upstream's
`GGML_CUDA_ALLREDUCE=internal` already has a 2-GPU chunked kernel: one launch per reduce,
staged through pinned host memory, with in-kernel cross-GPU sync on host flags. It was gated off
below Volta only by `__nanosleep` and a cc ≥ 700 init guard. Fork commit: sm60 busy-spin, guard
lowered to Pascal, and `GGML_PASCAL_AR_INTERNAL_MAX_BYTES` caps which reduces go internal. Set to
1048575 (just under upstream's 1 MB copy-engine threshold), only decode/verify reduces use the
kernel; prefill keeps the meta butterfly (P2P + F16 + duplex).

**Claim.** Decode rises with numerics unchanged; prefill flat.

**Test** (`scripts/run-h37-internal-ar.sh`, one lever: `GGML_CUDA_ALLREDUCE=internal` + cap):
1. Identity gate. Q4_K_M perplexity, 8 chunks, **-ub 16** so every reduce is ~327 KB and takes the
   chunked kernel. BF16 wire off (`GGML_CUDA_AR_BF16_THRESHOLD=0`): f32, two-term sum. Pass = mean
   KLD ≤ 1e-5 and same-top 100%.
2. Speed. Serve command at 16k (REPS=2) plus internal. Control `h36-duplex-16k` (40.37 / 365.3).
- 1-card arm skipped (rule zero): one card makes no reduces.

**Prediction.** Decode +5–15% (the ceiling is the ~12 ms/token of reduce gaps, ~22%). Prefill
within ±1%. Risk: a busy-spinning block steals an SM from the matvec on Pascal (no nanosleep).

**Result (2026-10-14, bench "H37 ... identity + 16k speed", fork dfcaa7f).** **REFUTED — not adopted.**
- Identity: internal vs none at -ub 16, mean KLD −0.000005, same-top 100% (13.75 vs 13.82 s/pass).
  No init-fallback warning in either log, so the internal pipeline came up.
- 16k speed, REPS=2: 40.15 / 40.16 t/s decode, 363.85 / 366.00 t/s prefill, 72.2–72.4% acceptance.
  Against `h36-duplex-16k` (40.37 / 365.3): **−0.5% decode, flat prefill**, inside noise. Peak 65 °C.
- Reading: replacing the reduce's copy+add with a single kernel did not move decode, so the
  ~100 µs matvec→add gaps are not transport time. They are most likely the meta backend's host-side
  per-subgraph dispatch (one subgraph boundary per reduce). The decode lever, if any, is in the
  meta scheduler, not the reduce kernel. Serve command unchanged.

## H38 — half2 Q·K accumulation in the FA tile kernel (fork plan P5)

**Why.** At 98k, prefill is 232 t/s against 365 at 16k. A linear fit to those two averages, and the
pp2048 trace (fattn 102 ms at ~1k depth), put attention at ~50% of per-token prefill time at 64k:
P5's 25% gate is met without a profile run. The sm60 prefill path is `fattn-tile`, D=256,
32 cols/block, `ncols2=2` (GQA 6). A one-card micro-bench (new test-backend-ops perf case: 12 Q / 2 KV
heads, kv 32k, nb 512) runs it at 37.4 ms, ≈5.5 TFLOPS, ~29% of fp16 peak.
- **Config sweep, closed** (fork `build-logs/p5-fa-sweep.sh`, 8 variants of nthreads/occupancy/
  nbatch_fa/nbatch_K): stock 256/2/64/64 is fastest. 256/2/64/32 +1.2%, 256/2/32/64 +4.7%,
  256/1/64/128 +8.6%, 512/1/64/64 +15%; 256/1/128/64, 128/2/64/64 and 256/2/128/32 are ~2.4× slower.
- **The lever:** the Q·K inner loop does `float += half22float2(hmul2)` (5 instructions per 4 flops),
  while V·KQ already uses `hfma2`. `GGML_PASCAL_FA_KQ_H2` accumulates `cpy_ne` (4) half2 products with
  `__hfma2`, then flushes to the fp32 accumulator once. Micro-bench: **37.4 → 27.8 ms (−26%)** f16 KV,
  37.7 → 28.5 ms q8_0 KV. All 151 D=256 FLASH_ATTN_EXT tests pass against CPU.

**Claim.** 100k prefill rises with quality held. Lossy: 8-product fp16 partial sums.

**Test** (`scripts/run-h38-fa-kq-h2.sh`, one lever, compile-time, `build-p5` vs `build-cuda-p100`):
1. KLD gate at depth: Q4_K_M, `-c 8192 --chunks 2` (scored tokens 4–8k deep), serve levers.
   Pass = mean KLD ≤ 0.001 (<2% of the Q4 quant step, H30 0.057) and same-top ≥ 99.5%.
2. 1-card IQ3_S, 16k, drafter-free, REPS=2. Control banked: `h34-1card-iq3` rep 2, 15.83 / 196.72.
3. 2-card serve command, 100k, REPS=1. Control banked: `serve-fork-100k`, 26.94 / 231.97.
- If adopted, the H32 100k haystack is re-run before the README command changes (lossy at depth).

**Prediction.** 100k prefill +8–12% (attention ~40% of the 0→98k average × 26%). 1-card 16k prefill
+3–5%. Decode at 100k +0–5% (MTP verify batches take the tile kernel).

**Result (2026-10-14, bench "H38 ... KLD gate FAILED", fork 5657e4b).** **REFUTED on quality — speed arms not run.**
- KLD at 4–8k depth, half2 vs fp32 Q·K accumulation: **mean 0.321**, median 0.000065, **max 29.5**,
  same-top 94.6%. Gate was ≤ 0.001 / ≥ 99.5%.
- The median is clean, so the damage is a tail. Upstream already scales Q by 0.25 on this path to keep
  fp16 products from overflowing; Qwen's outlier channels make 8-term fp16 partial sums lose all
  precision when large terms cancel. The 151 test-backend-ops cases passed only because their
  random inputs have no outliers. Not fixable by a shorter chunk without giving the speed back.
- Code removed from the fork. Follow-up: run the tile kernel's existing **fp32** path on P100
  (plain fp32 FMA, more precise than today's path, not less), micro-bench first.
- **Follow-up result (micro-bench only, no GPU serve run):** the fp32 tile path is **+14% slower**
  (f16 KV 42.55 vs 37.29 ms; q8_0 43.17 vs 37.73). Dropped. Cheap P5 levers are closed.

## H39 — staged, coalesced dequant-to-F16 before the prefill GEMM (fork plan P3 leftover)

**Why.** Phase 0 put the dequant pass at 5.8% of 2-card pp2048 and 7.3% of 1-card. The Phase 0 trace
gives its bandwidth: `dequantize_block_q4_K` on a 17408×2560 slice takes 732 µs, ~155 GB/s of
read+write; q5_K ~205, iq4_xs ~165, iq3_s ~115 GB/s. P100 streams ~500 GB/s. Two causes are visible
in the code. The 32-thread CTAs cap occupancy at 50% (sm_60 holds 32 CTAs per SM). Each thread also
stores 2-byte values at an 8-byte stride.

**Lever** (fork, `GGML_PASCAL_DEQUANT_STAGED=1`, runtime, off by default): each 256-thread CTA
dequantizes 256/nthreads super-blocks into shared memory with the stock device functions, then writes
them out as coalesced 16-byte stores. Same arithmetic, so the output is **bit-identical**. It covers
all K and IQ super-block types.

**Claim.** The dequant kernels run ≥ 2× faster, and prefill rises with output unchanged.

**Test** (one lever, same binary, env on/off):
1. `test-backend-ops -o MUL_MAT` correctness with the env on; identity check: PPL on/off equal to all
   digits (Q4_K_M, `-c 2048 --chunks 2`).
2. Micro-bench under nsys: dequant kernel time on/off. Skip the serve arms if under 1.5× faster.
3. 1-card IQ3_S, 16k, drafter-free, REPS=2. Control banked: `h34-1card-iq3` rep 2, 15.83 / 196.72.
4. 2-card serve command, 16k, REPS=2. Control banked: `h36-duplex-16k`, 40.37 / 365.3.

**Prediction.** 2-card 16k prefill +2.5–3.5%, 1-card +3–4.5%. Decode flat (it uses matvec, no dequant).

**Result (2026-10-14, bench "H39 staged dequant (identity + 1-card 16k + 2-card 16k)", fork 1aaab8b).**
**Identity PASS, speed marginal — code kept off by default, parked for the 220 W re-test.**
- Identity: KLD −0.000004, same-top 100% (the bit-identical noise floor of H36/H37).
- 1-card IQ3_S 16k: 199.43 / **205.56** prefill (control 196.72): rep 2 **+4.5%**. Decode 15.86 (flat).
- 2-card serve 16k: 352.12 / **370.11** prefill (control rep 1/2 364.17 / 366.40): rep 2 **+1.0%**,
  rep 1 −3.3%. Decode 40.2 (flat). Both arms' rep 1 came in low, unlike the controls (cold loads,
  191/215 s), so the 2-card result is inside noise.
- Power: the 1-card arm drew at the 175 W cap in 55% of busy samples (control 47%), so part of the
  kernel gain goes to clocks. See RUNBOOK "Re-test at the higher cap".
- Not added to the README command: the 2-card evidence does not clear noise. No re-run at 175 W.


## H40 — parallel host dispatch per card in the meta backend (decode, fork plan D-new)

**Why.** H37 showed the ~100 µs matvec→add gaps are not transport. The P2 step-0 trace (P1 + ALGO3 +
mmvh, `results/p2-step0`, CUDA API timeline, no new run) shows 2-card decode is **host-bound**. Over
64 tg tokens one host thread makes ~4,530 `cudaLaunchKernel` calls per token for both cards: ~25.6 ms
in the launch API plus ~15.8 ms of CPU between launches, against a ~54 ms traced token. That is
~9 µs of host per launch against ~20 µs of GPU per kernel. `ggml_backend_meta_graph_compute` issues
subgraph *i* for card 0 and then for card 1 from the same thread, so card 1 always starts a subgraph
late and card 0 idles at every reduce (~124 reduces per token).

**Lever** (fork, `GGML_PASCAL_PAR_DISPATCH=1`, runtime, off by default): one persistent worker thread
per extra card. The worker issues its card's subgraph while the main thread issues card 0's. They join
before the reduce, which stays serial. Same kernels, same order per stream, so output is
**bit-identical**.

**Claim.** 2-card decode rises because the host critical path per subgraph halves. Prefill is flat
(GPU-bound: launches are a small share at n=2048).

**Test** (one lever, same binary, env on/off):
1. Identity: Q4_K_M 2-card, `-c 2048 --chunks 2`, KLD on vs off. Pass = |KLD| ≤ 1e-5, same-top 100%.
2. 2-card serve command, 16k, REPS=2. Control banked: `h36-duplex-16k`, 40.37 / 365.3.
- No 1-card arm (rule zero): with one backend the meta path has no extra card and the code is
  inactive, so a 1-card arm would re-measure the control.

**Prediction.** 2-card 16k decode +8–15% with MTP (verify batches are launch-heavy too). Prefill ±1%.

**Result (2026-10-14, bench "H40 parallel dispatch (identity + 2-card 16k)", fork f24e1ff).** **REFUTED — code kept, off.**
- Identity: KLD −0.000004, same-top 100%; serve output identical to `h36-duplex-16k` (1413 chars).
- 2-card serve 16k: decode **40.55 / 40.48** (control 40.37, **+0.3%**), prefill 365.1 / 366.9 (flat).
- Follow-up trace (`results/h40-trace`, llama-bench tg32, dispatch on, ~1 min GPU): both cards are
  kernel-busy **89%** of the decode window. Launch cost is unchanged at ~6 µs per call on both threads,
  so the driver does not serialise them. The host-bound reading of the P2 step-0 trace was mostly CUPTI
  overhead (~2–3 µs added per launch). Untraced, one host thread already keeps up, and 2-card decode is
  GPU-bound. The remaining ~11% idle is reduce latency.
- **Decode host-dispatch work is closed** (graphs H27, one-shot reduce H37, parallel dispatch H40).
  Decode levers left are kernel-side: matvec speed and fusion. At 220 W the kernels shrink and the host
  share grows, so H40 is on the RUNBOOK re-test list.


## H41 — Q6_K mmvh with 4 lanes per superblock (decode, fork plan D2 follow-up)

**Why.** At the real per-card decode shapes (new perf cases, fork `test-backend-ops`, mmvh on,
card 0), the mmvh matvec runs at ~350 GB/s for Q4_K / Q5_K / IQ4_XS. Q6_K runs at ~190 GB/s:
8704×5120 takes 184 µs vs 70 µs for Q4_K. Q6_K is 11.5% of Q4_K_M decode bytes, including the 1 GB
output head, so it takes ~18% of matvec time. Cause: 210-byte blocks are 2-byte aligned. The 8-lane
split makes each lane do 32 two-byte loads for 32 weights, and every qh byte is fetched by 4 lanes,
every ql byte by 2.

**Lever** (fork, `GGML_PASCAL_MMVH_Q6K_WIDE=1`, runtime, off by default, needs `MMVH=1`): Q6_K uses
4 lanes per superblock. Each lane owns 16 positions of one 128-half (48 bytes, no overlap) and
produces 4 units of 16 weights. Loads are 5 aligned 4-byte words plus funnel shifts per 16 bytes.
Per-unit half2 math is unchanged; only the fp32 order of the per-lane sums changes.

**Micro (done, ~1 min GPU).** 8704×5120 184 → 110 µs, 5120×8704 193 → 106 µs, 5120×3072 71 → 46 µs
(−40%, ~330 GB/s). `test-backend-ops MUL_MAT q6_K`: all 38 n ≤ 8 cases pass, then the known
pre-existing `convert_nc` abort (H34).

**Test** (one lever, same binary, env on/off):
1. Decode-path KLD: Q4_K_M 2-card, `-c 2048 --chunks 2 -ub 8` (n ≤ 8, so mmvh runs), wide vs off.
   Pass = |mean KLD| ≤ 1e-4 and same-top ≥ 99.9% (fp32 reorder only).
2. 2-card serve command 16k, REPS=2. Control banked: `h36-duplex-16k`, 40.37 / 365.3.
- No 1-card arm (rule zero): UD-IQ3_S has 0.33 GB of Q6_K (~3% of bytes; its output head is Q5_K).
  Expected gain is ≤ 1%, below run noise, and H34 measured exactly that path as flat. Memory-bound,
  so it is not a 220 W re-test candidate either.

**Prediction.** 2-card 16k decode **+3–6%**. Prefill flat (n ≥ 512 never uses mmvh).

**Result (2026-10-14, logs `h41-off`/`h41-wide`, `h41-floor-*`, `h41-layer-*`).** **Gate FAILED, but the gate was wrong.**

| ub 8 KLD run | mean | max | p99.9 | same-top % | PPL (chunk 2) |
|---|---|---|---|---|---|
| wide vs off, `-sm tensor` | 0.002716 | 3.71 | 0.125 | 99.756 | 3.0372 vs 3.0478 |
| off vs off, `-sm tensor` (floor) | 0.000000 | 0.00005 | 0.00005 | 100.000 | — |
| wide vs off, `-sm layer` (no reduce) | 0.000711 | 0.58 | 0.138 | 99.560 | 3.0307 vs 3.0475 |

- Runs are deterministic (floor exact), and the reduce is not involved (`-sm layer` differs too).
- Kernel check (temporary `test-backend-ops` patch, fixed inputs, real shapes 8704×5120, 5120×8704,
  5120×3072 and 124160×5120 at n = 1 and 8): wide vs off max |Δ| ≤ 2.3e-5 on outputs of rms ~24,
  i.e. ~1e-6 relative. That is fp32 reordering only. NMSE vs CPU is identical to 4 significant figures
  for every case (e.g. 1.598e-05 both, 124160×5120 n=1).
- So this model at ub 8 turns a ~1e-6 rounding change into a KLD tail (median 1e-6, max up to 3.7,
  ~10–18 top-1 flips in 4094 tokens). Like H34/H35, the gate measured *difference*, not *loss*; a
  1e-4 bar is unreachable for anything short of bit-exact. PPL is lower, not higher, in both topologies.
- **Lesson:** decode-path (ub 8) KLD gates need a bit-exact identity check or a reference-based loss
  check (H35 style), never "wide vs off ≤ 1e-4" for a numerically equivalent change.

## H41b — Q6_K wide mmvh speed arm (re-gated)

**Gate (met, no GPU):** kernel-level equivalence to off at real shapes (above) and PPL not worse at
ub 8 in both topologies. No reference-based KLD run (rule zero): ~1e-6 relative fp32 reordering cannot
move the distance from exact arithmetic measurably next to mmvh's own ~4e-3 half-precision error,
and the NMSE vs CPU is unchanged.

**Run.** 2-card serve command 16k + `GGML_PASCAL_MMVH_Q6K_WIDE=1`, REPS=2. Control banked:
`h36-duplex-16k`, 40.37 / 365.3. No 1-card arm (H41 reason).

**Prediction.** Decode **+3–6%**, prefill flat. Adopt if decode ≥ +2% on both reps.

**Result (2026-10-14, `results/h11-placement.csv` `h41-wide-16k`).** **CONFIRMED, adopted.**
- Decode **41.48 / 41.54** (control 40.37, **+2.8%** both reps; adopt bar +2% met; prediction +3–6% just missed).
- Prefill 363.16 / 367.69 (flat). Acceptance 273/375 drafted (control 273/377–378): same accepted tokens, so
  the gain is kernel time, not drafting luck.
- Serve output diverges from `h36-duplex-16k` at char 546 (1413 vs 1357 chars), expected from the ub-8 sensitivity
  shown in H41.
- **Verdict:** `GGML_PASCAL_MMVH_Q6K_WIDE=1` joins the serve command. H32's 100k haystack gate is not re-run
  (rule zero): the change is a ~1e-6 relative fp32 reorder with unchanged NMSE vs CPU, three orders below the mmvh
  half-precision error that H32 already passed with.

## H42 — low-bit types on mmvh (1-card decode)

**Basis.** On UD-IQ3_S (1-card), Q3_K, Q2_K, IQ2_S, IQ2_XS, IQ2_XXS and IQ1_S still take the
mmvq path at ~83–115 GB/s. They are ~15% of decode weight bytes but ~31% of matvec time.
Fork e2f4a24 adds mmvh loaders for all but IQ1_S (0.3% of bytes, skipped), opt-in via
`GGML_PASCAL_MMVH_LOWBIT=1`. Micro-bench at 17408×5120, n=1, vs mmvq: Q3_K 353 → 179 µs,
Q2_K 265 → 101, IQ2_S 262 → 144, IQ2_XS 267 → 145, IQ2_XXS 253 → 132 (1.8–2.6×).
`test-backend-ops` MUL_MAT passes for every n ≤ 8 case of each type.

**Gate (H35-style, per the H41 lesson).** Reference: Q5_K_M on `build-h17` (sm_60 FP32),
2-card, `-c 2048 --chunks 2`. Two IQ3_S arms on the fork, 1 card, `-ub 8` (so decode
matvecs run), MMVH=1: `iq3-lowbit-off` (control) and `iq3-lowbit-on`. Pass: on-arm mean
KLD ≤ off-arm × 1.05 and same-top ≥ off-arm − 0.5 points.

**Speed arm** (if the gate passes, one lever): `h42-1card-iq3-lowbit` = H35's 1-card env
(MMVH=1, ALGO=1) + LOWBIT=1, 16k, REPS=2. Control: banked `h35-1card-iq3-algo3`
(15.84 / 198.5 mean). **Prediction: decode +8–15%, prefill flat** (n > 8 never uses mmvh).

**2-card arm: parked by rule zero.** Q4_K_M has only Q3_K (1.7% of bytes) among these
types: ≤ +1% expected, inside run-to-run noise at REPS=2. Re-open if LOWBIT is adopted for
1-card and the serve command is re-benched for another reason.

**Driver.** `scripts/run-h42-lowbit.sh`.

**Result (2026-10-14, `results/h42-kld.csv`, bench commit 934ad95).**

| vs Q5 FP32 reference, IQ3_S `-ub 8` 1-card | mean | p99 | p99.9 | same-top % |
|---|---|---|---|---|
| `iq3-lowbit-off` | 0.320269 | 10.21 | 25.67 | 91.251 |
| `iq3-lowbit-on` | 0.329384 | 11.04 | 26.01 | 91.251 |

Gate passed: +2.8% mean (≤ +5%), same-top unchanged. The shift is more than the ~1e-6 rounding
reorder H41 saw, and sits well inside the ±0.044 error bar. Likely cause: IQ2 weights carry a
+64 bias into the half2 products, so their fp16 partial sums are larger and lose more low bits.
Worth a look if LOWBIT ever goes into the 2-card command; not chased for 1-card.

| Speed arm (rep 1 / rep 2) | decode t/s | prefill t/s | vs `h35-1card-iq3-algo3` (15.84 / 198.5) | predicted |
|---|---|---|---|---|
| `h42-1card-iq3-lowbit` | 17.89 / 17.90 | 201.44 / 200.88 | decode **+13.0%**, prefill +1.3% (noise) | +8–15% ✓ |

**Verdict: adopt `GGML_PASCAL_MMVH_LOWBIT=1` for the 1-card arm.** The 2-card serve command is
unchanged (rule-zero park stands: Q3_K only, ≤ +1%).

## H43 — branch-free Q4_K/Q5_K scale unpack (decode)

**Basis.** A loads-only skeleton of the Q5_K matvec ran 365 → 489 GB/s at 8704×5120, so the main
K-quant matvecs are ALU/issue-bound, not memory-bound (fork plan, "Decode prize sizing"). Their
scale/min unpack branched on `is < 4`, which diverges across the 8 lanes of a superblock. Fork
aea9046 computes both forms from three aligned int loads and selects. The integer values are the same,
so the output should be bit-identical. Micro-bench vs the old code: Q4_K 61.4 / 66.2 / 28.2 µs (was
70.7 / 79.7 / 30.9; +10–20%), Q5_K 74.7 / 77.9 / 33.7 µs (was 83.3 / 90.6 / 36.3; +8–16%) at
8704×5120 / 5120×8704 / 5120×3072. MUL_MAT tests pass.

**Arm.** The serve command at 16k, REPS=2, on the new binary (no env change). Control: banked
`h41-wide-16k` (41.51 / 365.4).
- **Identity:** greedy output identical to `h41-wide-16k` (`compare-outputs.py`). If it differs,
  the change is not what it claims, so stop and investigate before reading the speed.
- **Prediction:** Q4_K + Q5_K are ~53% of decode weight bytes. ~13% off their matvec time is ~7% off
  all matvec time, and matvec is ~45% of a token. So **decode +2–4%**, prefill flat (n > 8 never uses
  mmvh).
- **1-card arm: parked by rule zero.** Q5_K + Q4_K are ~10.6% of UD-IQ3_S bytes, so ≤ +1% is expected,
  inside REPS=2 noise. The change is bit-identical, so there is no quality question to answer on 1 card.

**Driver.** `scripts/run-h43-k4scale.sh`.

**Result (2026-10-15, bench "H43 branch-free K-quant scales 2-card 16k").**

| Arm (rep 1 / rep 2) | decode t/s | prefill t/s | accept | vs `h41-wide-16k` (41.51 / 365.4) |
|---|---|---|---|---|
| `h43-k4scale-16k` | 42.42 / 42.36 | 359.95 / 366.37 | 72.8% | decode **+2.1%**, prefill flat |

Greedy output is identical to `h41-wide-16k` (1357 chars), with the same acceptance, so H32 still holds.
Inside the predicted +2–4%. **Verdict: adopted.** The change is in the binary with no flag, so the
serve command line is unchanged.

## H44 — Q3_K and IQ4_NL onto mmvh for the 2-card command (decode)

**Basis.** The P2 step-1 decode trace (bench 704213f) has two types still on mmvq in the 2-card command:
`mul_mat_vec_q<Q3_K>` 80.4 ms and `<IQ4_NL>` 45.1 ms of 1566 ms matvec, i.e. 8% of matvec time for 3.6% of
the bytes (UD-Q4_K_M: Q3_K 1.6%, IQ4_NL 2.0%). H42's `GGML_PASCAL_MMVH_LOWBIT=1` already puts Q3_K on mmvh
(adopted for 1 card only; 2-card parked by rule zero at "≤ +1%"). The trace now sizes that, so the arm is re-opened.
Fork 2d3b3bf adds IQ4_NL to the same flag, with aligned 4-byte loads (`mmvh_load16_b2<n>`; the 2-byte loads
made it a wash). Micro-bench vs mmvq at 8704×5120 / 5120×8704 / 5120×3072: Q3_K 98.9 / 98.4 / 44.3 µs
(was 180.3 / 168.2 / 76.2), IQ4_NL 81.6 / 102.1 / 37.7 µs (was 113.3 / 105.1 / 47.8). MUL_MAT tests pass.

**Gate (KLD).** Rounding changes (fp16 partial sums), so not bit-identical. Same method as H42, on 2 cards:
Q5_K_M FP32 reference (build-h17, `-sm tensor`), then UD-Q4_K_M with the full serve env at `-ub 8`
(the matvec path), LOWBIT off vs on. Pass: on mean KLD ≤ off × 1.05 and same-top ≥ off − 0.5.

**Arm.** The serve command at 16k + `GGML_PASCAL_MMVH_LOWBIT=1`, REPS=2. Control: banked `h43-k4scale-16k`
(42.39 / 366.4, same binary behaviour without the flag).
- **Prediction:** trace saving ≈ 45% of the Q3_K time + 25% of the IQ4_NL time ≈ 47 ms of 3705 ms
  drafter-free tg wall, so **decode +0.8–1.8%**, prefill flat.
- **1-card arm: parked by rule zero.** UD-IQ3_S has one IQ4_NL tensor (0.0% of bytes), and its Q3_K is
  already on mmvh under H42, so the 1-card command would not change measurably.

**Driver.** `scripts/run-h44-lowbit-2card.sh`.

**Result (2026-10-15, bench "H44 low-bit mmvh 2-card").**

| KLD vs Q5 FP32 ref (`-ub 8`, 2 chunks) | mean | p99 | p99.9 | same top |
|---|---|---|---|---|
| `q4-lowbit-off` | 0.126126 ± 0.029 | 1.760 | 22.50 | 95.015% |
| `q4-lowbit-on` | 0.124071 ± 0.029 | 1.749 | 22.67 | 95.015% |

Gate passed (on ≤ off; same top identical).

| Arm (rep 1 / rep 2) | decode t/s | prefill t/s | accept | vs `h43-k4scale-16k` (42.39 / 366.4) |
|---|---|---|---|---|
| `h44-lowbit-16k` | 44.95 / 44.96 | 361.13 / 366.39 | 72.8% | decode **+6.1%**, prefill flat |

Greedy output is identical to `h43-k4scale-16k` (1357 chars), with the same acceptance. **Verdict: adopted.**
`GGML_PASCAL_MMVH_LOWBIT=1` joins the serve env.

The gain is ~4× the prediction, because the prediction came from the drafter-free trace. The MTP drafter
(`mtp-Qwen3.8-27B-Q4_0.gguf`; the name is misleading) has **no Q4_0 tensors**. Its `output.weight`
(248320×5120) is **Q3_K, 546 MB**, read on every draft token. LOWBIT moves it from mmvq to mmvh. Lesson for
sizing: size decode levers from a trace *with* the drafter on, and check the drafter's tensor types.

## H45 — Drafter LM head trimmed to a vocab prefix

**Claim.** The drafter's `output.weight` (Q3_K, 546 MB, split 273 MB/card) is read on every draft token, about 9% of decode bytes. A head of the first K = 98,304 rows (216 MB, 108 MB/card) cuts that by 60%. Expected gain: up to about +5% decode.

**Why lossless.** The target verifies every draft token, so output must be byte-identical to `h44-lowbit-16k`. Only acceptance can fall: a draft can never be an id ≥ K. Offline tokenisation of code and Markdown (CPU, llama-tokenize) puts ids ≥ 96k at 0.01–0.6% of tokens; ids ≥ 64k were 2.5–2.9%, so K = 64k was rejected.

**How.**
- `scripts/pascal-trim-mtp-head.py` in the fork writes `blk.64.nextn.shared_head_head.weight` as the row prefix. The MTP graph already prefers that tensor.
- The fork loader accepts ne[1] ≤ n_vocab, and the tensor split puts it on axis 1, like `output.weight`.
- Drafter sampling runs on the backend (top-k 10 on a K-wide row), so ids need no remap. The raw-logits copy asserts a full-width head.

**Test.** 2-card 16k, REPS=3, serve env, the trimmed drafter swapped in. Control is banked: `h44-lowbit-16k` 44.96 / 366.4, acceptance 72.8% (2 reps).

**Gate.**
- Output identical to the control.
- Decode above 44.96 + noise.
- Acceptance no more than about 1 point lower.

**1-card arm:** parked by rule zero. The 1-card command (`h42-1card-iq3-lowbit`) runs with no drafter, so this lever can't touch it.

**Result (2026-10-15).** Confirmed and adopted.
- **First attempt** failed at load. The drafter context also copies raw logits (warmup/prompt), so the full-width assert fired. Fork fd90268 widens the K-wide rows on the host instead, with ids ≥ K set to −inf.
- **Rerun** (`h45-head96k-16k`): 46.31 / 46.38 / 46.35 decode, 363.4 / 367.4 / 366.6 prefill.
- **Gate:** output byte-identical to `h44-lowbit-16k`, and acceptance 273/375 exactly as before, so no draft on this prompt needed an id ≥ 96k.
- **VRAM:** +104 MiB/card. The unused full `output.weight` is still loaded; 4 GB is spare at 100k, so dropping it isn't worth a change.
- **Not tested, rule zero:** a smaller K. At 64k, 2.5–2.9% of tokens fall outside, which would cost more acceptance than the further ~1% byte saving.


## H46 — Prefill flash attention through cuBLAS GEMMs

**Claim.** At 98k depth, FA is ~40% of prefill. The sm_60 tile kernel runs at 5.5 TFLOPS. A cuBLAS micro-bench (fork `build-logs/p5-cublas-attn-bench.cu`) on the same shapes (2 kv heads/card, GQA 6, D 256, ubatch 2048, KV chunk 4096) runs Q·K with fp32 accumulate at 7.6 TFLOPS and P·V as an HGEMM at 14.5 TFLOPS. Expected gain: FA ~1.6× faster, 98k prefill +12–18%. Little to none at 16k.

**Precision.** It should match or beat the tile kernel. Q·K accumulates in fp32, as now; logits stay f32 through softmax. The tile kernel already accumulates P·V in half2, and the HGEMM does the same, but per chunk; chunks combine in f32. Q is converted to f16 as now.

**How.** Fork `ggml-cuda/fattn-pascal-gemm.cu`, used only when the env is set and the tile kernel is selected. Conditions: n_q ≥ 512, n_kv ≥ 4096, mask present, no sinks/ALiBi/softcap, D = DV. Everything else falls back to the tile kernel.

**Test.**
1. test-backend-ops FLASH_ATTN_EXT with the env set, against CPU.
2. KLD gate at depth, H38 style (`-c 8192 --chunks 2` vs env off): pass if mean ≤ 0.001 and same-top ≥ 99.5%.
3. Speed: 2-card, REPS=3, prefill at depth. A fresh control is needed because `serve-fork-100k` predates H41b–H45. 16k is checked only through the 1-card arm, since a 16k gain is expected to be small.
4. 1-card arm: IQ3_S, `-sm none`, at the same depth if it fits.

**Gate.** KLD passes; prefill at depth above control + noise; decode flat; the H32 100k haystack re-run before the README changes.

**KLD gate (2026-10-15): failed as pre-registered, then diagnosed as unmeetable.** GEMM vs tile at q8_0 KV: mean **0.320**, median 0.000066, max 33.0, same-top 94.0%.
- **Not a kernel bug.** A per-call tile-vs-GEMM diff on the real graph (temporary `GGML_PASCAL_FA_GEMM_CHECK`, 96 calls at kv 4096–8192) gave NMSE 3–8e-5 on average and 6.8e-4 at most. Making P·V more precise (fp32 accumulate) moved KLD *up* to 0.431.
- **The tail is the prompt, not the kernel.** All arms are scored against a non-FA reference (`-sm layer -fa 0 -ub 1024`, f16 KV, `logs/h46-diag-*`):

| Arm vs non-FA reference | mean KLD | median | max | same top |
|---|---|---|---|---|
| tile FA, `-sm layer` | 0.444 | 0.000084 | 36.5 | 93.32% |
| tile FA, `-sm tensor` (today's serve path) | 0.581 | 0.000101 | 36.0 | 92.63% |
| **GEMM FA, `-sm tensor`** | **0.555** | 0.000093 | 33.0 | **92.87%** |

- **Reading.** Any change to the attention arithmetic flips a handful of hypersensitive tokens in `h26-100k-prompt.txt` at 4–8k depth, each with a KLD of about 30. The mean KLD of an attention lever is therefore dominated by that tail. An absolute "≤ 0.001" gate can't be met even by the tile kernel vs non-FA.
- **H38 was refuted on this artifact.** Its 0.321 (median 0.000065, max 29.5) has the same shape. The H38 code is gone, and H46 is faster than H38 was, so H38 stays closed.
- **Re-specified gate (applied after the result, stated openly):** GEMM's KLD vs the non-FA reference must be ≤ tile's. It is (0.555 vs 0.581; same top 92.87 vs 92.63%). **Pass.** The H32 100k haystack before any README change stays as the end-to-end quality check.

**Speed (2026-10-15, `scripts/run-h46-fa-gemm.sh`, SKIP_KLD=1).**

| Arm | reps | decode t/s | prefill t/s | accept |
|---|---|---|---|---|
| 1-card IQ3_S 16k, control `h42-1card-iq3-lowbit` | 2 | 17.90 | 201.2 | — |
| 1-card IQ3_S 16k, `h46-1card-iq3-fagemm` | 3 | 18.02 | 205.2 (200.3 / 207.7 / 207.6) | — |
| 2-card serve 100k, control `h46-ctrl-100k` | 3 | 29.65 | 231.36 | 65.3% |
| **2-card serve 100k, `h46-fagemm-100k`** | 3 | 28.38 | **275.75** | 60.5% |

- **Prefill at 100k: +19.2%**, reps within ±0.05 t/s. At 16k on one card the gain is ~+3% (rep 1 low), as expected: attention is a small share of prefill at short depth.
- **Decode −4.3% is the text, not the kernel.** The GEMM path only fires for n_q ≥ 512, so decode and MTP-verify batches run the exact same kernels. Greedy output diverges from control at char 1066 (`compare-outputs`), and the new continuation drafts worse (257/425 vs 264/404 accepted). Same prompt, different words: a content effect, not a speed regression. The 10-rep headline will average this in.
- Speed gate: **pass**. Adoption waits on the H32 100k haystack (`scripts/run-h46-haystack.sh`).


## H47 — fp16-compute Q·K in the GEMM flash-attention path

**Claim.** In H46 the Q·K GEMM (fp32 compute, ~7.6 TFLOPS) is ~65% of attention GEMM time; P·V already runs as an HGEMM at ~14.5. Running Q·K as an HGEMM too (S stored f16, Q multiplied by the softmax scale before the f16 convert so |S| stays far below the f16 range) should cut FA time ~30%. At 100k, where attention is now ~30% of prefill, that would be roughly +8–10% prefill.

**Why it may be allowed now.** H38 dropped fp16 Q·K accumulation on a KLD of 0.321. H46's diagnosis showed that number is the prompt's hypersensitive-token tail, which any attention change hits (tile vs non-FA: 0.58). So the old refutation does not hold. The risk is still real: a 256-term fp16 sum carries an absolute logit error of about |logit|·2⁻¹¹·√256.

**Test.**
1. test-backend-ops FLASH_ATTN_EXT with the env set (thresholds at 1), against CPU.
2. Micro-bench (`test-backend-ops perf`, real per-card shape): continue only if FA time falls ≥ 15%.
3. KLD vs the banked non-FA reference (`/root/h28-kld/h46-nofa-base.kld`, `run-h46-diag.sh`): **pass if mean KLD ≤ 0.581 (tile, today's serve path) and same-top ≥ 92.63%**, i.e. no worse than tile.
4. Speed: 2-card 100k REPS=3 vs the banked `h46-fagemm-100k` (275.75); 1-card arm IQ3_S 16k REPS=3 vs `h46-1card-iq3-fagemm`.
5. If adopted: H32 100k haystack before the README changes.

**H32 haystack (2026-10-15, `scripts/run-h46-haystack.sh`):** `p8-h46-fagemm` **19/19** (reference 18/19, previous candidate 19/19). **Adopted** into the serve command.

**Result (2026-10-15).** Micro-bench at the real per-card shape: FA 48.5 → 34.6 ms (−29%, 11.9 TFLOPS); test-backend-ops 3993/3993. **KLD gate failed:** mean 0.618 (bar 0.581), same-top 92.30% (bar 92.63%), median 1.03e-4, max 32.8 (`logs/h46-diag-h47-qk16.log`). **Refuted** without a speed run. The fork code is committed for the record (8fa4400) and reverted (d3c1c12).

## H15 step 1 — 175 → 200 W (2026-10-15, user-set 200 W cap, `scripts/run-h15-200w.sh`)

2-card serve command (incl. H46), 16k, REPS=2 by user call (quick trajectory check; 98k after review).

| Cap | decode t/s | prefill t/s | peak temp | peak power gpu0 / gpu1 |
|---|---|---|---|---|
| 175 W `p15-175w-16k` | 47.47 (47.39 / 47.54) | 376.4 (375.1 / 377.7) | 65 °C | 178 / 192 W |
| **200 W `p15-200w-16k`** | 47.55 (47.49 / 47.61) | **393.5** (393.4 / 393.6) | 66 °C | 203 / 206 W |

- **Prefill +4.5%, decode flat (+0.2%)**, output identical (1224 chars). Heat: +1 °C peak.
- Side result: H46 at 16k on 2 cards (175 W) is 376.4 vs `h45-head96k-16k` 365.8, +2.9%.
- Cap restored to the user's 200 W after the control arm.

**Step 2 — 220 W (user-set, `scripts/run-h15-220w.sh`, controls banked above):** `p15-220w-16k` decode 47.63 (47.63 / 47.63), prefill **400.5** (398.7 / 402.4), peak 68 °C, peak power 221 / 213 W. Output identical.

| Cap | prefill | vs 175 W | step gain |
|---|---|---|---|
| 175 W | 376.4 | — | — |
| 200 W | 393.5 | +4.5% | +4.5% per 25 W |
| 220 W | 400.5 | +6.4% | +1.8% per 20 W |

- **Diminishing returns.** At 220 W card 1 peaks at 213 W, under the cap: at 16k the 2-card arm is mostly no longer power-limited, so 250 W would add little here. Decode is flat at every cap (+0.3% total).
- 98k (attention-heavy, longer sustained load) may still be cap-bound; next step is the user's call.

**Step 3 — 250 W (user-set "just for fun", `scripts/run-h15-250w.sh`):** `p15-250w-16k` decode 47.61 (47.56 / 47.65), prefill **400.8** (398.3 / 403.2), peak 69 °C, peak power 216 / 211 W. Output identical. Cap restored to 220 W (approved ceiling) after the run.

| Cap | decode | prefill | vs 175 W | peak temp | peak power gpu0 / gpu1 |
|---|---|---|---|---|---|
| 175 W | 47.47 | 376.4 | — | 65 °C | 178 / 192 W |
| 200 W | 47.55 | 393.5 | +4.5% | 66 °C | 203 / 206 W |
| 220 W | 47.63 | 400.5 | +6.4% | 68 °C | 221 / 213 W |
| 250 W | 47.61 | 400.8 | +6.5% | 69 °C | 216 / 211 W |

- **Flat above 220 W at 16k** (+0.1%, inside rep noise). Peak draw at 250 W never exceeds 216 W: the 16k serve load does not use the headroom. The 16k curve saturates at ~210-220 W.

**Step 4 — 98k at 200 W (user-set cap, REPS=2 by user call, `scripts/run-h15-200w-98k.sh`):** `p15-200w-98k` decode 28.44 (28.40 / 28.47), prefill **287.8** (287.82 / 287.83), peak 70 °C, peak power 213 / 209 W. Output identical to `h46-fagemm-100k` (1862 chars).

| 98k | decode | prefill | vs 175 W | peak temp |
|---|---|---|---|---|
| 175 W `h46-fagemm-100k` (REPS=3) | 28.38 | 275.75 | — | 67 °C |
| 200 W `p15-200w-98k` | 28.44 | 287.8 | **+4.4%** | 70 °C |

- 98k gains the same as 16k at 200 W (+4.4% vs +4.5%): the long prompt is not more cap-bound than the short one. 220 W at 98k not yet run (user's call).
