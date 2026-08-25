# The chunked gated-delta-net path — it already exists

**2026-08-25 · research only, no runs · follow-up to `prefill-ttft-2026-08-25.md`**

The previous research pass called a chunked GDN kernel "the largest lever we
control" and costed it as a from-scratch CUDA build project. **That costing was
wrong, and in our favour.** The chunked algorithm is already implemented in this
codebase, as a graph of generic ggml ops, and every op it needs is supported on
CUDA with no compute-capability gate. It is not selected on our rig because a
runtime flag prefers the fused kernel whenever the backend supports it — which
sm_60 does.

This turns the lever from *write a kernel* into *flip a cparam and measure*.

---

## 1. What the current kernel actually does

`ggml/src/ggml-cuda/gated_delta_net.cu`. The whole recurrence is one kernel, and
its structure is the finding:

```c
// line 63
for (int t = 0; t < n_tokens; t++) {
    ...
    float kv_col   = warp_reduce_sum<warp_size>(kv_shard);      // reduction 1
    float delta_col = (v_t[col] - g_val * kv_col) * beta_val;
    for (int r = 0; r < rows_per_lane; r++) {
        s_shard[r]    = g_val * s_shard[r] + k_reg[r] * delta_col;
        attn_partial += s_shard[r] * q_reg[r];
    }
    float attn_col = warp_reduce_sum<warp_size>(attn_partial);  // reduction 2
}
```

```c
// line 180, launch_gated_delta_net
//TODO: Add chunked kernel for even faster pre-fill
dim3 grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);   // num_warps = 4
dim3 block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);
```

**Four properties, each of which matters:**

1. **The grid does not contain `n_tokens`.** Parallelism is `H × n_seqs ×
   ceil(S_v/4)` blocks and nothing else. For our model (below) that is
   `48 × 1 × 32 = 1536` blocks of 128 threads, *whether the prefill is 512 tokens
   or 100,000*. Tokens are a sequential `for` loop inside each block. GDN time is
   therefore exactly linear in prompt length with a fixed, modest parallel width.
2. **Zero GEMM, zero FP16.** Every operation is scalar FP32 arithmetic plus warp
   shuffles. The 18.7 TFLOPS FP16 peak per card — the number the whole §2 floor
   argument is built on — is *entirely unused* by the 49 of 65 layers that run
   this kernel.
3. **Two serial warp reductions per token, and the second depends on the first**
   (reduction 2 consumes `s_shard`, which is updated using `delta_col`, which
   comes from reduction 1). Each token is a ~10-shuffle-deep dependency chain.
   This is a latency-bound scalar scan, not a throughput-bound numeric kernel.
4. **It is invariant to `-ub`.** Same total tokens, same per-token cost; only the
   launch count changes (49 launches per ubatch), which is microseconds. This is
   consistent with H14's +63% having come from the *GEMMs* getting taller, and it
   has an Amdahl consequence — see §5.

## 2. Our model's GDN dimensions

From the GGUF metadata of `Qwen3.8-27B-UD-Q4_K_M`:

| key | value | meaning |
|---|---|---|
| `qwen35.block_count` | 65 | total layers |
| `qwen35.full_attention_interval` | 4 | ~16 attention, **~49 GDN** |
| `qwen35.ssm.state_size` | 128 | `S_v` — GDN head dim |
| `qwen35.ssm.inner_size` | 6144 | value width → **`H_v` = 6144/128 = 48 heads** |
| `qwen35.ssm.group_count` | 16 | **`H_k` = 16 key heads** (48/16 = 3× GQA) |
| `qwen35.ssm.time_step_rank` | 48 | matches `H_v` |
| `qwen35.ssm.conv_kernel` | 4 | short conv before the recurrence |

`H_k = 16 ≠ H_v = 48` matters — see the caveat in §4.

**Scan FLOPs are a rounding error; scan *time* is not.** Per token per GDN layer
the recurrence is ~`H_v × S_v × S_v × 4` ≈ 3.1 MFLOP, so ~154 MFLOP/token across
49 layers — against ~54,000 MFLOP/token for the weight GEMMs. **The scan is ~0.3%
of prefill FLOPs.** If it turns out to consume a meaningful share of prefill
*time*, that is a statement that its efficiency is two to three orders of
magnitude below the GEMMs — which is exactly what §1's structure predicts.

## 3. The chunked path already exists — in graph form

`src/models/delta-net-base.cpp`:

```cpp
std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_delta_net(...) {
    if (n_seq_tokens == 1) { ... }                       // decode
    if (cparams.fused_gdn_ch) {
        return build_delta_net_fused(q, k, v, g, b, s, il);   // the §1 kernel
    }
    return build_delta_net_chunking(q, k, v, g, b, s, il);    // <-- chunked
}
```

`build_delta_net_chunking()` is the chunkwise-parallel (WY / UT-transform) form of
the delta rule, expressed in generic ggml ops: **10 × `ggml_mul_mat`**, plus
`ggml_solve_tri`, `ggml_tri`, `ggml_cumsum`, `ggml_exp`, `ggml_sum_rows`,
`ggml_pad`, and reshapes. Chunk size is **`CS = 64`** for the non-KDA path
(`delta-net-base.cpp:61`). This is the standard reformulation: an
`O(T · S²)` sequential scalar scan becomes `T/CS` sequential steps of dense
matmuls — the same transformation `ssm-scan.cu` applies to Mamba-2, whose comment
reads *"This converts the O(T*N) sequential scan into parallel matmuls."*

**It is not selected on our rig for a mundane reason.** `fused_gdn_ch` defaults to
`true` (`llama-context.cpp:233`) and is then auto-resolved by a probe that asks
only *does this backend support the fused op* (`llama-context.cpp:559-563`).
CUDA sm_60 supports it, so the probe says yes and the fused kernel wins. **There
is no CLI flag** — `fgdn`/`fused_gdn` appears nowhere in `common/arg.cpp`. Forcing
the chunked path is a ~2-line source change plus a rebuild.

### Every op it needs runs on sm_60

This was the thing most likely to kill the idea, and it does not.
`ggml-cuda.cu:5307-5311` returns `true` for `GGML_OP_CUMSUM`, `GGML_OP_TRI`,
`GGML_OP_DIAG` and `GGML_OP_SOLVE_TRI` **unconditionally — no `cc` comparison**.
`solve_tri.cu`, `tri.cu` and `cumsum.cu` contain no `GGML_CUDA_CC_*` gate at all.
`solve_tri.cu`'s fast path is `MAX_N_FAST 64`, which exactly matches `CS = 64`.

That matters because the ops-not-supported case is a real failure mode people hit:
[issue #24712](https://github.com/ggml-org/llama.cpp/issues/24712) is exactly the
"layer assigned to CPU but the fused GDN tensor is assigned to CUDA0 (usually due
to missing support)" warning. We should not hit it, but **watch the load log for
it** — a silent CPU fallback of the chunked graph would be catastrophically slow
and would look like a failed hypothesis rather than a misconfiguration.

## 4. The honest risks

**The Mamba-2 precedent is gated to Turing+, and nobody wrote down why.** The
directly analogous chunked path in `ssm-scan.cu:829` reads:

```c
const bool use_ssd = is_mamba2 && n_t > SSM_SSD_MIN_TOKENS && K == 1
                  && n_t <= SSM_SSD_MAX_TOKENS
                  && GGML_CUDA_CC_IS_NVIDIA(cc)
                  && cc >= GGML_CUDA_CC_TURING          // 750 — P100 is 600
                  && nr % 8 == 0;
```

introduced whole in [#22675](https://github.com/ggml-org/llama.cpp/pull/22675)
with only the comment *"Requires NVIDIA Turing+ otherwise fallback to scan."*
No stated reason. The two candidate explanations have opposite implications for
us: if it is a *correctness/API* constraint (that path stages `M_out` as `half`
for cuBLAS), it may not apply to the GDN graph path, which is FP32 throughout. If
it is a *performance* finding — that without tensor cores the matmul form loses to
the scalar scan — then **it is a direct prediction that this hypothesis fails on
P100**, and it is the single best reason to think so. This is the risk to hold in
mind, and the reason to measure rather than to build anything.

**The unfused path pays a 3× q/k expansion.** `qwen35.cpp:441`:

```cpp
// note: need explicit repeat only if we are not using the fused GDN.
if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
    q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
}
```

With `H_k = 16` and `H_v = 48` this materialises q and k at **3× width**. The
fused kernel handles the GQA broadcast implicitly (`neqk1_magic` / `fastmodulo`)
and pays nothing. Real cost, charged against whatever chunking saves.

**More ops, more launches, more intermediates.** ~50 ops per GDN layer instead of
one, × 49 layers × ubatches. VRAM for the intermediates scales with **ubatch, not
context** — at `-ub 2048`, `n_chunks = 32`, and a `CS × CS × n_chunks × H_v`
tensor is 64×64×32×48×4 B = **25 MiB**; a handful of those is a few hundred MiB.
Affordable, and crucially it does *not* grow toward 100k. But it competes with the
activation memory that H14 already found tight enough to OOM at `-ub 8192`.

**The FP32 ceiling.** These are FP32 GEMMs, so the relevant peak is 9.3 TFLOPS/card,
not 18.7. The chunked form also does ~1.5× *more* arithmetic than the scan. Both
are easily paid for if the scan is running two orders of magnitude below peak,
which is the whole premise — but it means the upside is bounded well short of the
naive "GEMMs are 100× faster" intuition.

## 5. What it is worth, and why H14 made it worth more

The scan is ~0.3% of prefill FLOPs, so its *time* share is entirely a question of
efficiency, and that is H18's measurement, not something to assert. What can be
said now:

**H14 raised the value of this lever rather than lowering it.** The GDN kernel is
invariant to `-ub` (§1.4), so H14's +63% came from the GEMM side alone. Amdahl
then says GDN's *share of the remaining time* went up: if it was 20% of prefill at
`-ub 512`, it is ~29% at `-ub 2048`; if it was 35%, it is now ~47%. Every
improvement to the GEMM path makes the sequential scan a bigger fraction of what
is left. **Run H18 at `-ub 2048` before estimating this, not at the old default.**

If GDN is ~30% of prefill and the chunked form is even 10× more efficient on it,
end-to-end prefill improves ~1.4×. That is the same 1.3–1.7× band the previous
research pass guessed — the number has not changed, but the *cost of finding out*
has dropped from a kernel-writing project to a patch and a benchmark.

## 6. Upstream — what exists and what does not

| | |
|---|---|
| [#19504](https://github.com/ggml-org/llama.cpp/pull/19504) | The op itself. Origin of the `//TODO: Add chunked kernel` we are looking at |
| [#22967](https://github.com/ggml-org/llama.cpp/issues/22967) | **RFC: chunked CUDA prefill kernel for `ggml_gated_delta_net`** — closed 2026-06-27. Confirms the gap and names `build_delta_net_chunking()` as the reference. Planned non-KDA first, dispatch chunked above a token threshold. **Closed without a CUDA PR landing** |
| [#24561](https://github.com/ggml-org/llama.cpp/pull/24561) | CUDA/HIP chunked MFMA prefill kernel — **CDNA (AMD) only**. Closed |
| [#20377](https://github.com/ggml-org/llama.cpp/pull/20377) | Vulkan chunked parallel GDN kernel — **still open** |
| [FlashQLA](https://qwen.ai/blog?id=flashqla) | 2–3× over the FLA Triton kernel for GDN chunked prefill — **Hopper SM90 only**. Already closed in the previous pass |

**So there is no chunked *CUDA* GDN kernel upstream, and the RFC to write one was
closed without landing.** Nobody is going to hand us this. But the graph-level
chunked path is already there and already correct — it is the default on every
backend that lacks the fused op, so it is well exercised for correctness, just
never for Pascal performance.

**Also worth knowing:** [#19345](https://github.com/ggml-org/llama.cpp/issues/19345)
reports llama.cpp ~40% slower than vLLM on Qwen Coder Next with high CPU usage —
the same architecture family, and consistent with GDN prefill being the weak spot
across the board, not something specific to our cards.

## 7. Proposed test — H22

Cheap, and it is a measurement rather than a build:

1. Patch `llama-context.cpp` to force `cparams.fused_gdn_ch = false` and
   `auto_fgdn = false` — ideally behind an env var (`LLAMA_GDN_CHUNKED=1`) so both
   paths are reachable from one binary and can be A/B'd without a rebuild.
2. Rebuild for sm_60.
3. **Check the load log for CPU-fallback warnings before trusting any number.**
4. `llama-bench -p 2048,4096,8192 -n 0 -ub 2048 -sm tensor`, fused vs chunked.
5. If prefill improves, re-check decode — `n_tokens == 1` takes the separate
   autoregressive path and should be untouched, but verify rather than assume.
6. Gate on quality: the chunked and fused paths are different arithmetic. Run one
   NIAH tier before quoting a throughput win.

The result is informative either way. A win is a large free speedup. A loss is the
Turing+ gate in `ssm-scan.cu` explaining itself, and it closes the last kernel-level
lever on the list — after which the remaining levers are the power cap and the
work-reducing ones (cache reuse, fewer tokens, smaller model).

---

## Sources

- Local: `ggml/src/ggml-cuda/gated_delta_net.cu`, `ggml/src/ggml-cuda/ssm-scan.cu`,
  `ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml/src/ggml-cuda/solve_tri.cu`,
  `src/models/delta-net-base.cpp`, `src/models/qwen35.cpp`, `src/llama-context.cpp`,
  `src/llama-cparams.h` — all at `/root/dflash2-rebased` (`57affa09`)
- GGUF metadata of `/root/Qwen3.8-27B-UD-Q4_K_M.gguf`
- [RFC #22967](https://github.com/ggml-org/llama.cpp/issues/22967) ·
  [PR #19504](https://github.com/ggml-org/llama.cpp/pull/19504) ·
  [PR #24561](https://github.com/ggml-org/llama.cpp/pull/24561) ·
  [PR #20377](https://github.com/ggml-org/llama.cpp/pull/20377) ·
  [PR #22675](https://github.com/ggml-org/llama.cpp/pull/22675) ·
  [issue #24712](https://github.com/ggml-org/llama.cpp/issues/24712) ·
  [issue #19345](https://github.com/ggml-org/llama.cpp/issues/19345)
- [FlashQLA (Qwen)](https://qwen.ai/blog?id=flashqla)
