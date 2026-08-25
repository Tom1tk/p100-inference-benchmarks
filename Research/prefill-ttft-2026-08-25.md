# Prefill and TTFT on dual Tesla P100 — research review

**Date:** 2026-08-25 · **Status:** research only, no runs executed · **Scope:** what can be
done about time-to-first-token and prefill throughput at 64k–100k context on this rig.

The brief: *"200 t/s at 64k context is over 5 minutes, our goal will be 100k."* This
document checks that premise against what we have actually measured, works out the
hardware floor, finds where the time is really going, surveys every engine and technique
that could apply, and ends with the hypotheses worth spending runs on.

Everything below is either measured on this rig, read out of the source tree we build
from, or cited. Claims that are none of those are labelled as estimates.

---

## 1. Correcting the premise

The 200 t/s figure is real but it is a **16k** number, and it is the only depth we have
ever measured with a real prefill. Two separate gaps:

**Gap 1 — every llama-bench prefill number in this repo was taken at `n_depth = 0`.**
`run-bench.sh` passes `-p 0,2048,4096,8192,16384`, which sets *prompt length*, not depth.
`n_depth` stayed 0 in all 16 raw CSVs. So we have measured "prefill a prompt of length N
into an empty cache", never "prefill with N tokens already resident". For a prefill-cost
question that is the right measurement, but it means we have no data past 16,384 tokens.

**Gap 2 — every prefill number was taken at `n_ubatch = 512`, `n_batch = 2048`.** These
are the defaults. We have never swept them. See H14.

What we do have, `mainline-rebased`, `-sm tensor`, `UD-Q4_K_M`:

| prompt tokens | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|
| prefill t/s | 222.6 | 223.6 | 215.0 | 208.4 |

**Prefill is almost flat from 2k to 16k — only −6.4%.** In a pure transformer you would
expect a visible quadratic bend by 16k. There isn't one. Section 3 explains why, and it
is the most important finding in this document.

### What 100k probably costs today

We have never run the 27B past 16k, so this is bracketed, not measured. Two bounds:

- **Optimistic** (cost stays linear, 208 t/s holds): 97,201 / 208 = **467 s ≈ 7.8 min**.
- **Pessimistic** (apply the decay curve measured on this rig for Qwen3.5-9B in the July
  NIAH runs — 8k→100k throughput fell to 47.3% of its 8k value): ~102 t/s →
  **953 s ≈ 15.9 min**.

At 64k the same bracket gives **5.0 – 8.1 min**. So the brief's "over 5 minutes at 64k"
is the *optimistic* end of the range. The concern is well founded and probably
understated. **Measuring the actual curve is the first thing to do (H13)** — the whole
plan below branches on which end of that bracket we are on.

### The July NIAH data we already own

`/root/niah_test/` holds a complete 8k/32k/64k/100k sweep from 2026-07-16 that nobody has
cited since. It used **Qwen3.5-9B-UD-Q4_K_XL**, not the 27B, so it is not a baseline for
the current model — but it is the only long-context scaling data measured on these cards:

| tier | prompt_n | buun prefill t/s | TTFT | decode t/s |
|---|---|---|---|---|
| 8k | 7,839 | 396.1 | 19.8 s | 22.6 |
| 32k | 31,139 | 304.5 | 102.3 s | 11.9 |
| 64k | 62,223 | 234.7 | 265.1 s | 7.8 |
| 100k | 97,201 | 187.9 | 517.3 s | 5.6 |

Two things worth flagging. First, **decode collapses with depth** — 22.6 → 5.6 t/s, a 4×
loss, on a 9B. The plan has been optimising decode at 4k–16k; at 100k decode is a
different problem than the one we have been solving. Second, NIAH retrieval was **12/12
hits** for both engines at every tier and depth, so the harness works and the fixtures are
good.

The paired `pflash_turbo3` rows are +14% to +19% faster than buun at every tier. That is
*not* PFlash working: `run_engine.py` passes no `--pflash-*` flags at all. It is the
pflash *build* with turbo3 KV, nothing more. **PFlash's prefill acceleration has never
been exercised on this rig.** Section 5 explains why it can't be.

---

## 2. The hardware floor

Worth establishing before chasing speedups, because it bounds every option.

Prefill is compute-bound: roughly `2 × N_params` FLOPs per token, so
`2 × 27e9 = 54 GFLOP/token` for this model.

- 2× Tesla P100-PCIE-16GB, FP16 peak **18.7 TFLOPS each = 37.4 TFLOPS** aggregate at the
  stock 250 W.
- Measured today: 208 t/s × 54 GFLOP = **11.2 TFLOP/s**, i.e. **30% of stock peak**.
- Absolute floor for a 100k prefill at 100% of peak: 5.4 PFLOP / 37.4 TFLOP/s = **144 s**.
- A realistic 45–55% of peak would be **260–320 s ≈ 4.5–5.5 min**.

**So a dense 27B cannot prefill 100k on this rig in under ~2.5 minutes, and ~5 minutes is
a good outcome.** Anything materially faster requires doing *less work*: prefilling fewer
tokens, using a smaller model, or not prefilling at all (cache reuse). That is the single
most useful sentence in this document and it should shape which hypotheses get run first.

### The cards are power-capped at 175 W of a 250 W default

```
0, Tesla P100-PCIE-16GB, 175.00 W, default 250.00 W, max 250.00 W, clocks.max.sm 1328 MHz
1, Tesla P100-PCIE-16GB, 175.00 W, default 250.00 W, max 250.00 W, clocks.max.sm 1328 MHz
```

That is a 30% power cut on a workload that is compute-bound and therefore clock-bound.
Prefill should respond to this almost linearly until thermals bind; decode, being
bandwidth-bound, should barely move. Raising the limit is deferred per standing
instruction and collides with the 83 °C rule, but it is now clear that **it is a prefill
lever specifically**, not a general one. See H15 — it needs an explicit decision.

### Memory is not the constraint at 100k

The model is GQA with 4 KV heads × 256 key/value length, and only ~16 of 65 layers carry
a KV cache (see §3):

`16 layers × (4×256 + 4×256) × 2 B = 64 KiB/token` → **6.25 GiB at 100k, f16.**

With 16.5 GB of weights that is ~22.8 GB of 31.8 GB usable VRAM. **100k at f16 KV fits
comfortably.** KV quantisation is therefore not needed to reach 100k, which matters
because H4 already showed turbo KV costs 14.3% of decode. Host RAM is the tighter
resource: **28 GB total, ~27 GB available** — that bounds the prompt-cache lever in H19.

---

## 3. Where the prefill time actually goes

This is the finding that reframes the problem.

### The model is a hybrid, not a transformer

GGUF metadata for `Qwen3.8-27B-UD-Q4_K_M`:

```
qwen35.block_count             = 65
qwen35.full_attention_interval = 4
qwen35.ssm.conv_kernel         = 4      qwen35.ssm.state_size    = 128
qwen35.ssm.group_count         = 16     qwen35.ssm.inner_size    = 6144
qwen35.attention.head_count    = 24     qwen35.attention.head_count_kv = 4
qwen35.attention.key_length    = 256    qwen35.attention.value_length  = 256
```

`full_attention_interval = 4` means **only ~16 of 65 layers are full attention. The other
~49 are gated-delta-net (linear attention) layers.** This explains the flat 2k→16k curve
in §1 directly: three quarters of the network costs O(N), not O(N²).

It also means most sparse-attention research does not apply here. Techniques that attack
the quadratic term can only ever touch a quarter of the layers.

### The gated-delta-net CUDA kernel walks tokens one at a time

`ggml/src/ggml-cuda/gated_delta_net.cu`, the kernel that runs those ~49 layers:

```c
// line 63
for (int t = 0; t < n_tokens; t++) {
    ...  // warp reductions per token, sequential state update
}
```

and at the launch site, line 180:

```c
//TODO: Add chunked kernel for even faster pre-fill
```

The op was added in [PR #19504](https://github.com/ggml-org/llama.cpp/pull/19504) as a
**vector implementation, with the chunked implementation explicitly left as future work.**
For a 512-token ubatch that is 512 sequential steps per layer, 49 layers deep, with only
warp-level reductions for parallelism — no GEMMs, so no use of the P100's 2:1 FP16 rate.

The contrast is stark: `ssm-scan.cu` in the same tree *does* have the chunked treatment
(`SSM_SSD_CHUNK_SIZE 256`, "converts the O(T*N) sequential scan into parallel matmuls",
`SSM_SSD_MIN_TOKENS 128` to switch over for prefill). The Mamba-2 path got a prefill
kernel. The gated-delta-net path did not.

**Working thesis: prefill on this model is dominated by the sequential GDN kernel, not by
the quantised GEMMs.** That would explain (a) the flat length curve, (b) why we sit at
only 30% of FP16 peak, and (c) a corroborating data point from §5.1 — the sm_60 bug report
measured that forcing the main GEMM path from FP16 to FP32 compute leaves *prefill
throughput unchanged*. If halving the GEMM rate does not move prefill, prefill is not
GEMM-bound.

This is a thesis, not a measurement. **H18 tests it**, and it should be tested early
because it decides whether the rest of the plan is aimed at the right target.

### What a chunked GDN kernel would be worth

Published results on the chunked formulation, for scale:

- Qwen's **FlashQLA**: 2–3× forward speedup on GDN chunked prefill — **Hopper SM90+ only**,
  requires TMA and warpgroup MMA.
- **Netra Runtime**: 0.91 ms → 0.31 ms (2.9×) for 32k prefill at TP8 on H200.
- A Blackwell llama.cpp fork reports 9.5× on a GDN microbenchmark.

The *algorithm* is architecture-neutral: chunk of 64 tokens, invert
`A = (I − tril(βKKᵀ))⁻¹` per chunk via triangular solve, then dense matmuls, one state
hand-off between chunks. Nothing there needs tensor cores — on P100 you would issue cuBLAS
HGEMM instead of wgmma. What *is* Hopper-only is the warp-specialised producer/consumer
schedule that gets the last chunk of the published speedup.

Honest estimate for a naive chunked port on sm_60: the algebraic and GEMM-conversion win
without the scheduling win, so **plausibly 1.5–2.5× on the GDN layers**. If those layers
are ~60% of prefill, that is roughly **1.3–1.7× end-to-end**. This is a real engineering
project (a new CUDA kernel), not a config change — but it is the largest lever we
control, and it is upstream-shaped work that would benefit every Pascal user.

---

## 4. Verified facts about our CUDA path (read from source, not assumed)

Several plausible-sounding knobs turn out to be already-settled at the source level. These
are worth recording so nobody spends a run on them.

**MMQ (integer matmul) is unavailable on P100 and always will be.** `mmq.cu:316`:
```c
if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) { return false; }   // 610
```
P100 is cc 600. `__dp4a` needs 6.1. So every quantised prefill GEMM goes
dequantise → cuBLAS. `GGML_CUDA_FORCE_MMQ` cannot change this. **Closed.**

**FP16 cuBLAS compute is already on.** `ggml-cuda.cu:1626`:
```c
compute_type = fast_fp16_hardware_available(cc) ? GGML_TYPE_F16 : GGML_TYPE_F32;
```
`fast_fp16_hardware_available` is true for cc 600 (it excludes only 610), and
`prefer_f32_output` is set only for cc == 700 exactly. So we already get
`CUBLAS_COMPUTE_16F`. There is no `GGML_CUDA_F16` win to collect — the flag isn't even in
our CMakeCache because it no longer gates this path. **Closed.**

**A new env knob exists that we have never used:** `GGML_CUDA_CUBLAS_COMPUTE_TYPE`
(`f32|f16|bf16|auto`) forces the compute type at runtime. Useful as the *measurement
instrument* for H18 — set it to `f32` and see whether prefill moves at all.

Full set of CUDA env knobs in our tree, for reference:
`GGML_CUDA_ALLREDUCE`, `GGML_CUDA_CUBLAS_COMPUTE_TYPE`, `GGML_CUDA_DEVICES`,
`GGML_CUDA_DISABLE_FUSION`, `GGML_CUDA_ENABLE_UNIFIED_MEMORY`, `GGML_CUDA_GRAPH_OPT`,
`GGML_CUDA_NO_PINNED`, `GGML_CUDA_P2P`, `GGML_CUDA_REGISTER_HOST`,
`GGML_OP_OFFLOAD_MIN_BATCH`, `GGML_SCHED_DEBUG`.

**The fused GDN kernel is active in our runs.** [PR #26177](https://x.com/TeksEdge/status/2082162711472054338)
(in b10152) fixed `--fit` miscounting the MTP block, which could silently disable the
fused GDN kernel and cost ~10% of decode. We build at `57affa09` / build 1183, well after
that, and no `sched_reserve … fused Gated Delta Net` warning appears in any of our server
logs. The two logs that matched a grep for `sched_reserve` were DFlash2 crash backtraces,
not this warning. **Not our bug — but worth a standing grep, since we don't pass `--fit`.**

---

## 5. Engine and technique survey

### 5.1 The sm_60 FP16 fast-path correctness bug — [issue #25593](https://github.com/ggml-org/llama.cpp/issues/25593)

**The most consequential external finding, and it is not about speed.**

sm_60 is routed into the FP16 fast path by default, and some math that should stay FP32 is
silently truncated to FP16. sm_61 already has a carve-out; sm_60 does not. Measured impact:

| | unpatched | patched |
|---|---|---|
| median KL divergence | 0.004962 | 0.000001 |
| same-top-token rate | 95.00% | 99.89% |

**About 1 in 20 greedy tokens flips.** The fix is three lines — add `cc != 600` alongside
the existing `!= 610` checks. Cost: **prefill unchanged, decode +1.4%.** So it is free.

Three consequences for us:

1. Our `mainline-rebased` and `mainline` builds are **unpatched**. Every quality
   observation we have made on them is affected.
2. **`buun` merged the fix** (its PR #80), as did `llama-cpp-turboquant` (PR #212). So
   `buun` and `rebased` have not been running the same arithmetic. Any cross-engine
   *quality* comparison in this repo is confounded; throughput comparisons are fine, since
   the fix is throughput-neutral on prefill.
3. It bears directly on **"PFlash was too lossy"**. That verdict was formed on P100 with
   this bug live. It does not overturn the verdict — 1-in-20 token flips is not the same
   failure mode as mangled tool calls — but it means the baseline those judgements were
   made against was itself degraded. Worth knowing before re-testing anything on quality.

That the fix leaves prefill throughput *identical* while halving the GEMM compute rate is
also the strongest external corroboration of the §3 thesis.

### 5.2 TurboPrefill — the most promising directly-applicable find

[sergey-automation/TurboPrefill](https://github.com/sergey-automation/TurboPrefill),
"Intra-Prompt Pipeline Scheduling for Multi-GPU Prefill", llama.cpp b10335 (2026-08-10).

Instead of running each ubatch through all layers before starting the next, it keeps
several ubatches in flight at different pipeline stages, filling the bubbles that
layer-split multi-GPU inference leaves.

| rig | model | baseline | TurboPrefill | speedup |
|---|---|---|---|---|
| 12× P104-100 | Llama-3-70B | 37 t/s | 199 t/s | **5.3×** |
| 4× RTX 3090 | Llama-3-70B | 400 t/s | 1208 t/s | 3.0× |
| 8× RTX 5060 Ti | GPT-OSS-120B | 1963 t/s | 4380 t/s | 2.23× |

**Pascal is a validated target.** Constraints that matter to us:

- **`-sm layer` only.** Tensor split "works but is bandwidth-limited". This directly
  collides with our best config: H6 established `-sm tensor` wins, and Phase 2 measured
  tensor+MTP at 29.26 t/s vs layer+MTP at 19.05 t/s decode — **layer costs 35% of decode.**
- Needs `n_batch` 4–64× `n_ubatch` (e.g. B=4096, UB=128) and context ≥ 4× ubatch.
- **Prefill only** — decode path untouched.
- Speedup scales with pipeline depth, and **we have 2 GPUs, not 12.** The 5.3× came from a
  12-deep pipeline. A 2-deep pipeline should be expected to yield **~1.3–1.6×**, not 5×.

So it is a genuine TTFT lever with a known decode cost, which makes it a clean tradeoff to
measure rather than an obvious win. See H16.

### 5.3 PFlash — architecturally unavailable

[lucebox PFlash](https://www.lucebox.com/blog/pflash) (2026-05-01) claims 10.4× TTFT at
128k on a 3090: a Qwen3-0.6B drafter scores token importance across the whole prompt, the
27B target prefills only the surviving spans (`keep_ratio` default 0.05), with
Block-Sparse-Attention accelerating the drafter's own long-context forward.

**It requires sm_80+.** The four hand-written kernels (`mean_K → score → select →
sparse_fwd`) plus BSA explicitly target Ampere. **P100 cannot run it.** This closes the
"is there a newer PFlash worth trying" question: there is no v2, and the shipped v1 is
architecturally out of reach regardless of the earlier quality verdict.

Reported quality, for the record: NIAH single-needle passes at 32k/64k/128k at
`keep_ratio=0.05`, though third-party testing found 0.05 unreliable in practice and
[considers 0.10 the realistic floor at 128k](https://note.com/zephel01/n/n1cc5c4a9daeb?hl=en).

**But the idea is portable, and that is the interesting part.** The expensive,
sm_80-dependent piece is `sparse_fwd` — making the *drafter's* forward pass sparse. The
selection stage (per-block mean of K, per-(Q-block, K-block) scoring, top-k) is ordinary
SIMT arithmetic. A 0.6B drafter running a *dense* forward over 100k tokens is cheap in
absolute terms; we would forgo the drafter speedup and keep the target speedup, which is
where nearly all the win is. At `keep_ratio=0.10` the target prefills 10k instead of 100k
tokens. Even allowing generous overhead this is the only technique surveyed with a
plausible path to a **sub-2-minute 100k TTFT** on this hardware. It is also the riskiest
and the most work. See H21, and note it is gated on NIAH, which we already have.

### 5.4 Other engines

| Engine | sm_60? | Verdict |
|---|---|---|
| **vLLM** | No — requires cc ≥ 7.0 | Forks exist: [vllm-pascal](https://github.com/cduk/vllm-pascal) is **discontinued**, succeeded by [pascal-pkgs-ci](https://github.com/sasha0552/pascal-pkgs-ci), which tops out at **vLLM 0.10.0** and is "soft-broken due to PyTorch". vLLM 0.10 long predates `qwen35` hybrid-SSM support, and a 27B would need a 4-bit path whose fast kernels (Marlin) are sm_80+. **Closed.** |
| **SGLang** | No | sm_75+ minimum, same architecture gap. **Closed.** |
| **TensorRT-LLM** | No | Volta+, now effectively Ampere+. **Closed.** |
| **ExLlamaV2 / V3** | V3 is sm_80+ | V2 tolerates Pascal but has no hybrid-SSM/GDN support and no path to this model. **Closed.** |
| **MLC-LLM / TVM** | Compiles for sm_60 | No `qwen35` hybrid support; we would be writing GDN kernels in TVM instead of CUDA, with worse tooling. Third-party reports put MLC prefill *behind* llama.cpp even on supported hardware. **Not worth it.** |
| **ktransformers** | No | Ampere+. **Closed.** |
| **tinygrad** | Nominally, via its CUDA backend | Explicitly asked about, so stated plainly: tinygrad is a ~25-op compiler stack aimed at clarity and training, not a production inference engine. It has no GGUF loader, no gated-delta-net op, no hybrid-SSM model support, no quantised Pascal kernels, and no tensor-parallel serving. Reaching parity would mean reimplementing the entire stack we already have, then beating hand-tuned cuBLAS on a 10-year-old architecture its codegen does not target. **Recommend against.** |
| **ik_llama.cpp** | Yes | Already in the matrix. `-sm graph` only; no `qwen35`-specific prefill work. H12 remains the open question and is low value. |
| **buun / turboquant** | Yes | Already in the matrix, **and carries the sm_60 correctness fix** (§5.1) that mainline lacks. That is a new reason to keep it around. |

### 5.5 Techniques that reduce prefill work

| Technique | Applicable? | Note |
|---|---|---|
| **Prompt / prefix cache reuse** | **Yes — free** | `-cram/--cache-ram` (default only 8192 MiB), `--cache-idle-slots`, `-ctxcp`, `-cms`. Eliminates prefill entirely on a stable prefix. Bounded by 28 GB host RAM. **Highest practical value for the agentic loop.** See H19. |
| **ubatch/batch tuning** | **Yes — free, untested** | Never swept. See H14. |
| **Power limit 175→250 W** | Yes, gated | Prefill is clock-bound. See H15. |
| **Pipelined multi-GPU prefill** | Yes | TurboPrefill, §5.2. |
| **Chunked GDN kernel** | Yes, but it's a build | §3. Largest lever we control. |
| **Speculative prefill** | Partially | §5.3 / H21. |
| **Sparse attention (MInference, XAttention, FlexPrefill)** | Marginal | All target the quadratic term, which here is ~16 of 65 layers. All ship sm_80+ kernels. Low ceiling, high cost. |
| **Smaller model** | **Yes** | User explicitly opened this. See H20. |
| **KV quantisation** | Doesn't help prefill | Already refuted for decode (H4). Not needed for 100k memory either (§2). |
| **Chunked prefill (interactivity)** | No | Reorders work, doesn't reduce it. |

---

## 6. What this changes

1. **The target is misidentified.** We have been optimising decode via split mode and
   drafters. At 64k–100k, TTFT is minutes and decode is single-digit t/s. Both need work,
   and the decode-at-depth collapse (22.6 → 5.6 t/s on a 9B) is a problem the plan has
   never addressed.
2. **Prefill is probably not GEMM-bound**, so quant choice, MMQ, and FP16 flags are all
   dead ends. The sequential GDN kernel is the suspect.
3. **There is no 10× available at fixed work.** The floor is ~2.5 min for 100k. Getting
   materially under ~5 min means prefilling fewer tokens or a smaller model.
4. **The cheap wins are configuration, not engines**: ubatch, prompt-cache reuse, and
   possibly the power cap. None have been tried.
5. **We are running numerically degraded** relative to buun, for free, and it confounds
   every quality comparison we have made.
6. **PFlash is closed as a product** and **open as a technique**.

---

## 7. Proposed hypotheses

Full text in `HYPOTHESES.md`. Ordered by (value ÷ cost).

| ID | Claim | Cost |
|---|---|---|
| H13 | 27B prefill holds ≥150 t/s at 100k (decays less than the 9B did, because 49/65 layers are linear) | 1 run — **do first** |
| H14 | Raising `-ub` 512→1024/2048 improves prefill ≥10% | 1 sweep |
| H17 | The sm_60 FP16 fast-path fix is free (prefill flat, decode +1.4%) and changes output | 3-line patch + rebuild |
| H18 | GDN layers, not GEMMs, dominate prefill (>50% of a 16k prefill) | instrumentation |
| H19 | Prompt-cache reuse cuts agentic turn-2+ TTFT by >90% | 1 server config |
| H15 | 175→250 W lifts prefill ≥15%, decode <3% | gated on approval |
| H16 | TurboPrefill nets a TTFT win at 64k despite forcing `-sm layer` | build + matrix |
| H20 | A 9B on one P100 beats the 27B's TTFT by ≥2× at 64k at acceptable NIAH quality | 1 run, data partly exists |
| H21 | PFlash-style token selection is portable to sm_60 without the BSA kernels | spike — largest upside, largest risk |

---

## 8. Closed — do not spend runs

| Line | Why |
|---|---|
| PFlash as shipped | sm_80+ kernels; no v2 exists |
| vLLM / vllm-pascal / pascal-pkgs-ci | vLLM 0.10 ceiling, soft-broken, no `qwen35` |
| SGLang, TensorRT-LLM, ExLlamaV3, ktransformers | sm_75/sm_80+ minimum |
| MLC-LLM / TVM | no hybrid-SSM support; slower than llama.cpp where measured |
| tinygrad | not an inference engine; whole stack would need writing |
| FlashQLA | Hopper SM90 + TMA only |
| `GGML_CUDA_FORCE_MMQ` | dp4a needs cc 6.1; P100 is 6.0 (`mmq.cu:316`) |
| `GGML_CUDA_F16` | already effectively on (`ggml-cuda.cu:1626`) |
| KV quantisation for prefill or for reaching 100k | doesn't affect prefill; f16 KV fits 100k in VRAM |

---

## Sources

- [llama.cpp issue #25593 — SM_60 FP32 math silently done in FP16](https://github.com/ggml-org/llama.cpp/issues/25593)
- [TurboPrefill — multi-GPU prefill acceleration](https://github.com/sergey-automation/TurboPrefill)
- [llama.cpp PR #19504 — add GATED_DELTA_NET op](https://github.com/ggml-org/llama.cpp/pull/19504)
- [PFlash — 10× prefill at 128K on RTX 3090](https://www.lucebox.com/blog/pflash) · [repo](https://github.com/Luce-Org/lucebox-hub/tree/main/pflash) · [third-party keep_ratio testing](https://note.com/zephel01/n/n1cc5c4a9daeb?hl=en)
- [Qwen FlashQLA — CP/Bwd-friendly fused linear attention kernels for GDN](https://qwen.ai/blog?id=flashqla)
- [Netra Runtime — how to speed up GDN kernels for Qwen models](https://netraruntime.com/blog/how-to-speed-up-gdn-kernels-for-qwen-models)
- [llama.cpp discussion #27164 — Qwen3.8-27B / MTP GGUF issues](https://github.com/ggml-org/llama.cpp/discussions/27164)
- [llama.cpp issue #24712 — fused Gated Delta Net assigned to CPU warning](https://github.com/ggml-org/llama.cpp/issues/24712)
- [vllm-pascal](https://github.com/cduk/vllm-pascal) · [pascal-pkgs-ci](https://github.com/sasha0552/pascal-pkgs-ci) · [vLLM issue #963 — cc <7.0](https://github.com/vllm-project/vllm/issues/963)
- [tinygrad](https://github.com/tinygrad/tinygrad)
- [AI-assisted Gated DeltaNet optimization on NVIDIA Blackwell](https://arxiv.org/pdf/2607.16831)
- Local: `/root/niah_test/{buun,pflash}_results.jsonl` (2026-07-16), `results/raw/*.csv`,
  `dflash2-rebased/ggml/src/ggml-cuda/{gated_delta_net.cu,mmq.cu,ggml-cuda.cu,ssm-scan.cu}`
