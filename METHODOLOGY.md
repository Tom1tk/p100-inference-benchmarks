# Methodology

Hardware, engines, models, and fixed parameters. Change nothing here without
recording why in [RUNLOG.md](RUNLOG.md).

---

## 1. Objective

Compare three llama.cpp-family inference engines on identical hardware, model,
and quants to find the fastest stable configuration for serving Qwen3.8-27B on
the dual-P100 rig.

Primary metrics: prefill (pp) throughput across context depths up to 16k, and
decode (tg) throughput, with and without MTP speculative decoding.

This supersedes informal numbers from the prior single-GPU P100 session
(`Qwen3.5-27B-UD-Q3_K_XL`, 10.36 tok/s) and the claude.ai handover doc — those
are prior knowledge to verify, not baselines to assume.

---

## 2. Hardware

| | |
|---|---|
| Host | HP Z640, Proxmox, inference in an LXC |
| GPUs | 2× Tesla P100-PCIE-16GB (16269 MiB each, 32GB total), device 0 + 1 |
| Arch | GP100, compute capability 6.0 (sm_60) |
| Bandwidth | 732.2 GB/s HBM2 per card (the widely-quoted 549 GB/s figure is the 12GB variant) |
| Interconnect | PCIe 3.0, **no NVLink** — this is why NCCL AllReduce loses to direct P2P here |
| Power | 175W/card cap, confirmed applied. **Approved ceiling 220W** (H15, gated on a manual PSU check). Persistence mode enabled. Neither survives a host reboot |
| Cooling | Passive cards, custom 3D-printed shroud + Arduino Nano PWM controller |
| Thermal limits | Idle ≤45°C · load 70–80°C · throttle 83°C · hard shutdown 87°C |

### sm_60 constraints — the dominant architectural fact

No dp4a, no tensor cores, no async copy instructions. Most modern quantized MMQ
kernels assume sm_61+ and fall back to slow generic paths or fail to compile.

GP100 does have a native **2:1 FP16 rate (~19 TFLOPS)**, unusual for Pascal.
Paths that dequantize to f16 and hit cuBLAS perform well here; integer-quant
paths generally do not. This asymmetry underlies several hypotheses.

---

## 3. Engines under test

| Engine | Lineage | Commit | Binary directory |
|---|---|---|---|
| `pflash` | `Tom1tk/mtp-pflash-turboquant-hip` (HIP-descended, CUDA-capable) | `e05ff58b7` (build 9110) | `/root/pflash-llama.cpp/build-cuda-p100/bin/` |
| `buun` | `spiritbuun/buun-llama-cpp` (CUDA-native turboquant lineage) | `39d97a876` (build 11260) | `/root/buun-llama-cpp/build-cuda-p100/bin/` |
| `ik` | `ikawrakow/ik_llama.cpp` | `8337e4c` ("Fix Qwen35+ MTP") | `/root/ik_llama.cpp/build/bin/` |
| `mainline` | `ggml-org/llama.cpp` **PR #27342** (`dflash2` branch) | `64f765f5` | `/root/dflash2-llama.cpp/build-cuda-p100/bin/` |

`mainline` was added on 2026-08-24 specifically to run DFlash2, which none of
the three forks can (see H8). It is a `git worktree` of `/root/mainline-llama.cpp`
on the fetched PR branch, so `master` stays clean and the PR can be refreshed
with a plain `git fetch`:

```bash
cd /root/mainline-llama.cpp
git fetch origin pull/27342/head:pr-27342
git worktree add /root/dflash2-llama.cpp pr-27342
```

The PR is **open, not merged** (checked 2026-08-24: `state: OPEN`,
`mergeable: MERGEABLE`, base `master`, 16 files, +563/−7). It is small and
self-contained. Re-check before treating any `mainline` number as reproducible
from a released build — if it merges, pin the merge commit here instead.

### Split-mode support — verified, differs per engine

Checked directly against each binary's argument parser. The handover doc's
table was wrong about `ik`.

| Engine | `none` | `layer` | `row` | `tensor` | `graph` |
|---|---|---|---|---|---|
| `pflash` | ✅ | ✅ | ✅ | ✅ | ❌ |
| `buun` | ✅ | ✅ | ✅ | ✅ | ❌ |
| `ik` | ✅ | ✅ | ❌ | ❌ | ✅ |
| `mainline` / `mainline-rebased` | ✅ | ✅ | ✅ | ✅ | ❌ |

`-sm graph` exists **only** in `ik_llama`. Passing a value the binary doesn't
know gives `error: invalid parameter for argument: -sm`.

**Correction (2026-08-24): `-sm tensor` works, and it won Phase 1.** This
section previously said it was arch-gated off for `qwen35` and advised "don't
spend a run on `-sm tensor`". That was wrong. `llm_arch_supports_sm_tensor()`
is a **denylist**, not an allowlist:

```cpp
switch (arch) {
    case LLM_ARCH_GROK: ... case LLM_ARCH_QWEN3TTS:
        return false;
    default:
        return true;      // qwen35 lands here
}
```

Verified identical in both `mainline` (`src/llama-arch.cpp`) and `buun`
(`src/llama-arch.cpp:1023`). `qwen35` is absent from the list, so it is
**permitted**. `-sm tensor` is now the selected split mode — see Phase 1.

**`-sm tensor` requires `GGML_CUDA_ALLREDUCE=none` on this rig.** The NCCL
default aborts (H5, rig-wide); `internal` silently falls back to the same
butterfly path `none` uses, because it needs sm70+ (H10 Finding 3). Use `none`.

### Engine capability matrix — what can be combined (2026-08-25)

The three features worth combining do not all live in the same tree:

| Engine | `-sm tensor` | MTP | DFlash2 | TurboQuant KV |
|---|---|---|---|---|
| `mainline-rebased` | ✅ | ✅ | ✅ (layer only) | ❌ |
| `ik` | ❌ | ✅ (`--spec-type mtp`) | ❌ | ❌ |
| `buun` | ✅ | ✅ | ❌ | ✅ |
| `pflash` | ✅ | — | — | ✅ (excluded from testing) |

**`buun` is the only tree with tensor + MTP + TurboQuant together.**
`mainline-rebased` is the only tree with DFlash2 — and DFlash2 is layer-only
anyway (H8), so it can never be combined with tensor split regardless of engine.

TurboQuant KV types live in `buun` and `pflash` only: `turbo2`, `turbo3`,
`turbo4`, `turbo8`, `turbo3_tcq`, `turbo2_tcq`, `turbo1_tcq`, `vbr`. Neither
`ik` nor `mainline` has them, so H4 and H7 can only be tested on `buun`.

⚠️ **`buun`'s default KV type is `vbr` (implicit t4 floor), not `f16`.** Any
`buun` run that does not pin `-ctk`/`-ctv` is not measuring what a `mainline`
run with the same flags measures. `scripts/run-bench.sh` pins both to `f16`,
so existing rows are comparable — but hand-run `llama-server` invocations are
not, unless pinned.

### Dual-GPU tuning knobs — surveyed 2026-08-24

Beyond `-sm`, each engine exposes inter-GPU controls that this project has not
yet touched. Surveyed directly from `--help` and source; **none of these are
tested yet** — see H10–H12.

**Common to `pflash` / `buun` / `mainline`** (all llama.cpp-descended):

| Flag | What it does | Why it matters here |
|---|---|---|
| `-ts, --tensor-split` | Fraction of model per GPU | Both cards are identical, but the drafter and KV sit unevenly — an asymmetric split may beat 50/50 |
| `-mg, --main-gpu` | Which GPU holds non-split tensors | Relevant to H2's XL split-stall — the embedding/output head land here |
| `-ot, --override-tensor` | Pin tensor-name patterns to a device/buffer | Manual placement of the tensors XL can't split. **Also silently disables pipeline parallelism** — see below |
| `-dev, --device` | Restrict which devices are used | Single-GPU runs without `CUDA_VISIBLE_DEVICES` |
| `-devd, --device-draft` | **Device for the draft model** | Drafter on one card vs split across both — directly relevant to DFlash2/MTP |
| `-otd, --override-tensor-draft` | Per-tensor placement for the drafter | Finer version of the above |

**`mainline`-only, runtime env vars** (no rebuild needed):

| Env var | Default | Note |
|---|---|---|
| `GGML_CUDA_P2P` | **unset = peer access OFF** | These two cards *can* peer (verified `cudaDeviceCanAccessPeer = 1` both ways, PHB topology, same NUMA node). Mainline does not enable it unless asked |
| `GGML_CUDA_ALLREDUCE` | `nccl` on Linux | Also accepts `internal` (a dedicated **2-GPU** pipeline) and `none` (meta-backend butterfly). See H10 |
| `GGML_CUDA_CUBLAS_COMPUTE_TYPE` | `auto` | Auto already picks F16 for quantized mat-muls on fast-FP16 hardware, which P100 is. Forcing `f32` is a quality-vs-speed probe, not a win |
| `GGML_CUDA_DISABLE_GRAPHS` / `GGML_CUDA_GRAPH_OPT` | graphs on | `n_past` changes per decode step, so graph capture may not help — `ik`'s known-good baseline passed `-cuda graphs=0` |
| `GGML_CUDA_ENABLE_UNIFIED_MEMORY` | off | Would allow oversubscribing VRAM at PCIe speed. Escape hatch for models that don't fit, not a speed knob |
| `GGML_CUDA_REGISTER_HOST` / `GGML_CUDA_NO_PINNED` | pinned on | Host-transfer tuning; matters most during load |
| `GGML_CUDA_DISABLE_FUSION` | fusion on | Diagnostic |
| `GGML_CUDA_PDL` | off | Programmatic dependent launch — Hopper-era, irrelevant on sm_60 |
| `GGML_SCHED_DEBUG=1` | off | Prints the graph split assignment. **Diagnostic of record** for proving H2 |

**`ik`-only, graph-split tuning** — these exist because `-sm graph` needs
inter-GPU data exchange, and are unexplored:

| Flag | Default | Note |
|---|---|---|
| `-smf16` / `-smf32` / `-grt` | f16 | Precision of data exchanged between GPUs. P100 has 2:1 FP16 and no NVLink, so halving PCIe traffic is plausibly a real win |
| `-gap` | f16 | Flash-attn precision under `-sm graph` |
| `-smgs` | 0 | Force split-mode graph scheduling |
| `-sas` | 0 | Async evaluation of compute graphs |

**Pipeline parallelism (mainline/pflash/buun) is automatic and easy to lose.**
From `src/llama-context.cpp`:

```cpp
bool pipeline_parallel =
    model.n_devices() > 1 &&
    model.n_gpu_layers() > model.hparams.n_layer_all &&
    model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
    cparams.offload_kqv &&
    !model.has_tensor_overrides();
```

So it is **off** under `-sm none`, `-sm row`, `--no-kv-offload`, or *any* `-ot`
override. That last one is a trap: adding `-ot` to hand-place a tensor silently
costs pipeline parallelism, and the log doesn't shout about it. Any `-ot`
experiment must be compared against a no-`-ot` control, not against the Phase 1
baseline.

### MTP flag syntax — differs per engine

Verified from each engine's `--help`. Do not assume a shared syntax:

| Engine | Flag |
|---|---|
| `pflash` | `--spec-type mtp --spec-draft-n-max N` |
| `buun` | `--spec-type draft-mtp --spec-draft-n-max N` (enum value is `draft-mtp`, **not** `mtp`) |
| `ik` | `--spec-type mtp:n_max=N,p_min=P` |
| `mainline` | `--spec-type draft-mtp` (same enum as `buun`) |

### DFlash2 flag syntax (`mainline` only)

```bash
--model-draft /root/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
--spec-type draft-dflash --spec-draft-n-max 7
```

**The flag is the same `draft-dflash` used for v1.** PR #27342 does not add a
new `--spec-type` value — it auto-detects v2 from the draft GGUF's
`dflash.selector_top_k` metadata key (`common/speculative.cpp`:
`is_dflash2 = selector_top_k > 0`).

This is exactly why a v1 engine cannot be trusted to report a DFlash2 result:
the same command line runs a different algorithm depending on whether the loader
understands the selector. Verify from the server log that the selector was
picked up before recording any number.

### Other per-engine differences

- `buun`'s `-fa` is documented as `<on\|off\|auto>` but accepts `1`/`0` as
  well — verified, so `-fa 1` is portable across all three.
- `ik`'s `-fa` defaults to `1`; `pflash`/`buun` default to `0`/`auto`. Always
  pass it explicitly.
- `ik` accepts `-cuda graphs=0`, which the handover doc's known-good baseline
  used for graph split (`n_past` changes every decode step, so graph capture
  doesn't apply).

### Build commands

```bash
# pflash, buun
cmake -B build-cuda-p100 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda-p100 --config Release -j12

# ik_llama — NOTE: GGML_NCCL defaults ON and breaks graph split on this rig
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 -DCMAKE_BUILD_TYPE=Release -DGGML_NCCL=OFF
cmake --build build --config Release -j12

# mainline PR #27342 (DFlash2). LLAMA_CURL=OFF — no libcurl dev headers here,
# and models are all local anyway.
cd /root/dflash2-llama.cpp
cmake -B build-cuda-p100 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build-cuda-p100 --config Release -j12
```

**Pascal is still first-class in mainline.** `ggml/src/ggml-cuda/CMakeLists.txt`
documents `60 == P100, FP16 CUDA intrinsics` in its architecture list, and the
CUDA-12 branch of that list is what our toolkit (12.0) takes. The only new CUDA
kernel the PR adds (`ggml/src/ggml-cuda/top-k.cu`, a two-stage tiled top-k) uses
nothing newer than shared memory and `__syncthreads()` — no warp-level
reductions requiring sm_70, no tensor cores, no async copy, no bf16. Reviewed
before building, and it configured for `sm_60` without complaint.

The current `ik` build was made **without** `-DGGML_NCCL=OFF` and therefore
has NCCL linked in. This is the open blocker — see H5.

---

## 4. Models

### The target model is a hybrid, not a transformer — this drives everything

`Qwen3.8-27B` (`general.architecture = qwen35`) GGUF metadata:

```
qwen35.block_count             = 65     qwen35.full_attention_interval = 4
qwen35.ssm.state_size          = 128    qwen35.ssm.inner_size          = 6144
qwen35.ssm.group_count         = 16     qwen35.ssm.conv_kernel         = 4
qwen35.attention.head_count    = 24     qwen35.attention.head_count_kv = 4
qwen35.attention.key_length    = 256    qwen35.attention.value_length  = 256
```

`full_attention_interval = 4` means **only ~16 of 65 layers are full attention;
the other ~49 are gated-delta-net (linear attention).** Consequences:

- **Prefill is near-linear in prompt length**, not quadratic — measured −6.4%
  from 2k to 16k. Do not reason about this model with transformer intuitions.
- **KV is small**: 16 layers × (4×256 + 4×256) × 2 B = **64 KiB/token** →
  6.25 GiB at 100k, f16. **100k fits in VRAM without KV quantisation.**
- **Sparse-attention techniques have a low ceiling here** — they can only touch a
  quarter of the layers.
- The ~49 GDN layers run `gated_delta_net.cu`, whose CUDA kernel walks tokens
  **one at a time** (`for (int t = 0; t < n_tokens; t++)`, line 63) with an
  author `//TODO: Add chunked kernel for even faster pre-fill` at line 180.
  This is the prime suspect for the prefill bottleneck (H18).


Target: **Qwen3.8-27B** — 27.32B dense params, 262k native context,
architecture string `qwen35`, multimodal.

All files present on disk under `/root/`:

| File | Quantizer | Size | Role |
|---|---|---|---|
| `Qwen3.8-27B-UD-IQ3_S.gguf` | Unsloth Dynamic (IQ) | ~11 GiB (downloading) | **Single-GPU quant** — the only one that fits one 16 GiB P100 with room for KV |
| `Qwen3.8-27B-UD-Q4_K_M.gguf` | Unsloth Dynamic | 15.33 GiB | Split-friendly, large KV budget |
| `Qwen3.8-27B-UD-Q5_K_M.gguf` | Unsloth Dynamic | 18.41 GiB | Midpoint |
| `Qwen3.8-27B-UD-Q6_K_M.gguf` | Unsloth Dynamic | 21.50 GiB | **Primary test quant** — handover doc's recommended split target |
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` | Unsloth Dynamic (XL) | 16.35 GiB | **H2 split-stall test only** — not a general split candidate |
| `Qwen3.8-27B-Uncensored-noMTP-Q4_K_M.gguf` | non-Unsloth ("stock") | 15.41 GiB | H3 stock-quant degradation test. Filename implies no MTP head — confirm before use |
| `mtp-Qwen3.8-27B-Q4_0.gguf` | — | 1.28 GiB | MTP draft head. Verified: `qwen35.nextn_predict_layers = 1`, arch matches base |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | — | 1.06 GiB | DFlash2 drafter. Verified v2 — see below |
| `Qwen3.8-27B-DFlash2-Q8_0.gguf` | — | 1.92 GiB | DFlash2 drafter. Verified v2 — see below |

**`UD-IQ3_S` is not just another point on the quant curve.** At ~11 GiB it is
the only target that fits a *single* 16 GiB card alongside a usable KV cache, so
it is the model that makes `-sm none` measurable at all. Every other quant on
this list fails to load on one card (`Q6_K_M` at 21.5 GiB is what produced the
`failed to load model` in the Phase 1 table). It therefore does double duty:
a row in the Phase 3 quant sweep, and the single-GPU reference that H6 needs.

Expect it to be the slowest per-token of the dynamic quants on prefill — IQ
quants are more compute-heavy to dequantize, and sm_60 has no dp4a — so read its
Phase 3 numbers as "what single-GPU costs," not as a like-for-like quant
comparison against the K-quants.

### Drafters (speculative decoding)

Three drafters are under test, plus a no-drafter control. All target
Qwen3.8-27B and pair with **any** quantisation of it — a drafter is trained
against a base model, not against a target quant.

| Drafter | Source | Engines that can drive it | Status |
|---|---|---|---|
| _(none)_ | — | all | Control |
| MTP head | `mtp-Qwen3.8-27B-Q4_0.gguf` (verified) | pflash, buun, ik, mainline | Ready |
| DFlash2 Q4_K_M | `z-lab/Qwen3.8-27B-DFlash2` (1.06 GiB) | **`mainline` only** | On disk, v2 verified |
| DFlash2 Q8_0 | `z-lab/Qwen3.8-27B-DFlash2` (1.92 GiB) | **`mainline` only** | On disk, v2 verified |

Both DFlash2 files were confirmed to be genuine v2 by metadata, not by filename:

```
dflash.selector_rank  = 256
dflash.selector_top_k = 16
selector_hidden.weight / selector_predecessor.weight / selector_successor.weight
```

Those three tensors are exactly what PR #27342 adds mappings for
(`LLM_TENSOR_DFLASH_SELECTOR_{HIDDEN,PREV,NEXT}`). A v1 engine has no mapping
for them.

**Speculative decoding here is output-preserving.** The target model verifies
every drafted token over its full vocabulary, so drafter choice and drafter
quantisation affect *acceptance rate* — i.e. speed — and cannot change what the
target emits. Verified against the engines' flags: none of them expose a
relaxed/lenient acceptance mode. `--spec-draft-p-min` gates *drafting*, not
acceptance.

Consequence for the Q4-vs-Q8 drafter comparison (H9): it is a **speed and VRAM**
comparison, not a quality one. A worse drafter proposes worse tokens, the target
rejects more of them, and throughput drops — output does not degrade.

The one caveat: DFlash drafters read the *target's hidden states*. Those differ
between target quants, so a heavily quantised target can lower acceptance. Still
a speed effect, not a correctness one.

Verify MTP head compatibility without loading the full model:

```bash
/root/benchmarks/benchvenv/bin/gguf-dump --no-tensors <model>.gguf | grep -i nextn
```

### Quant rules carried from prior work (unverified — see H2, H3)

- **Don't use `-XL` quants on the dual-GPU split.** XL upcasts the embedding
  and output head to Q8_0; those can't be split, so they land on one GPU and
  stall the other during prefill. XL is claimed *better* on a single GPU.
- **If Qwen3.8 retains Gated Delta Network layers, don't drop below Q6 with
  stock quants** — `ssm_out` degrades disproportionately. Unsloth Dynamic
  quants are claimed exempt.

---

## 5. Explicitly out of scope

**PFlash and DFlash v1 are excluded from all testing.** Both have already been
evaluated on these cards and found too lossy to be useful — DFlash v1 also
regularly mangled tool calls. This is an empirical finding, not a theoretical
concern, and is not being re-litigated for those two.

`pflash` and `buun` bundle these schemes alongside TurboQuant/MTP in the same
binary — this means don't pass the `--spec-type` values that select them
(`dflash`, `draft-dspark`, etc.), not that the repos are off-limits.

**DFlash2 is explicitly back in scope** (2026-08-24, operator's call). It is a
different drafter from DFlash v1 — a separate checkpoint, a separate upstream PR
(#27342 vs #22105), and a candidate-path selector v1 doesn't have. The v1
verdict does not transfer to it. It is now runnable via the `mainline` engine
built from that PR; see H8.

**In scope:** TurboQuant (KV-cache quantization), MTP, and DFlash2.

---

## 6. Fixed parameters

Tool: each engine's built-in `llama-bench`. Not the server — avoids HTTP/JSON
overhead noise, and all three trees ship it. (Phase 2 MTP runs are the
exception; `llama-bench` can't drive speculative decoding.)

| Parameter | Value | Rationale |
|---|---|---|
| `-ngl` | `99` | Full offload |
| `-fa` | `1` | `-fa 0` aborts on sm_60 in graph-split mode. Used everywhere for consistency |
| `-t` | `8` | Thread count found irrelevant in prior work (6/8/10/12 within noise — fully GPU-bound). Fixed, not re-tuned |
| `-ctk` / `-ctv` | `f16` | f16 KV was previously "mandatory" (q4_0 cost 31–58%). TurboQuant KV types tested separately per H4 |
| `-r` | `3` | Mean ± stddev as `llama-bench` reports natively. Smoke tests used `-r 1` and are marked as such |
| `-p` | `0,2048,4096,8192,16384` | Prompt **lengths** to 16k — *not* depths. `llama-bench` takes depth via `-d`, which we have never used, so `n_depth` is 0 in every raw CSV. See "Prefill is measured at length, not depth" below. `-p 0` yields no output row — expected |
| `-n` | `128` | tg128, standard comparison point |
| `-o` | `csv` | Machine-readable. Aggregated into `results/all-results.csv` |
| `CUDA_VISIBLE_DEVICES` | `0,1` | Both P100s. Re-confirm enumeration each session — the chassis has been reconfigured before |

Canonical command:

```bash
CUDA_VISIBLE_DEVICES=0,1 <engine-bin>/llama-bench \
  -m <model>.gguf \
  -ngl 99 -fa 1 -t 8 \
  -ctk f16 -ctv f16 \
  -sm <split-mode> \
  -p 0,2048,4096,8192,16384 -n 128 -r 3
```

`scripts/run-bench.sh` wraps this — prefer it over running by hand, since it
also handles telemetry, logging, and the commit/push protocol.

### Prefill is measured at length, not depth — and only at `-ub 512`

Two limits on every prefill number in this repo, both discovered 2026-08-25.

**1. Length, not depth.** `-p N` prefills an N-token prompt into an *empty*
cache. `-d N` prefills with N tokens already resident. We have only ever passed
`-p`, so `n_depth` is 0 in all 16 raw CSVs. For "what does a cold prefill of N
tokens cost", which is the TTFT question, `-p` is the correct measurement — but
it means **we have no data at all past 16,384 tokens**, and nothing about the
cost of extending an existing context.

**2. `-ub 512`, always.** Every number was taken at the defaults `-ub 512` /
`-b 2048`. The ubatch/batch sweep has never been run (H14). Do not describe any
existing prefill figure as tuned.

When quoting prefill, state the length and the ubatch. "208 t/s" alone is
ambiguous and has already been extrapolated to 64k and 100k in conversation,
where the honest bracket is 5.0–8.1 min and 7.8–15.9 min respectively.

### Prefill has a hardware floor — check it before chasing a speedup

Prefill is compute-bound at roughly `2 × N_params` FLOPs per token: **54
GFLOP/token** for the 27B. Two P100s peak at **37.4 TFLOPS FP16** (18.7 each,
stock 250 W). So:

- 100k prefill at 100% of peak: **144 s**. Nothing beats this.
- At a realistic 45–55% of peak: **260–320 s**.
- Measured today: 208 t/s = **11.2 TFLOP/s = 30% of peak**.

Any proposal claiming a large TTFT win at 100k must either reduce the token
count prefilled, shrink the model, or skip the prefill (cache reuse). Kernel and
flag tuning is bounded by the gap between 30% and ~55% of peak — worth roughly
1.5–2×, not 10×.

### The cards are power-capped at 175 W of a 250 W default

```
0, Tesla P100-PCIE-16GB, 175.00 W, default 250.00 W, max 250.00 W
1, Tesla P100-PCIE-16GB, 175.00 W, default 250.00 W, max 250.00 W
```

This is a prefill-specific tax: prefill is clock-bound, decode is
bandwidth-bound.

**The ceiling is 220 W per card, approved 2026-08-25 — not 250 W.** The 175 W
setting was chosen as a conservative default, not derived from a measurement. All
runs to date were taken at 175 W and that stays the baseline; 220 W is the top of
the range H15 is allowed to explore, and 250 W stays out of bounds.

**One gate before any run above 175 W:** the wall draw of the PSU has never been
measured. The user checks it with a plug-socket power meter, in person — it is not
readable from this host — and until that check is reported, the cap stays at
175 W. When cleared, step 175 → 200 → 220, one step per run.

**Never change the cap silently — record it as a column**, and re-apply it after
any host reboot, which reverts both cards to the 250 W default.

### MTP measurement (Phase 2)

Effective tg t/s = tokens generated ÷ wall time. **Also record the acceptance
rate** where the engine reports it — the comparison target is a community
reference of ~1.7× on Qwen3.6-27B on a 3090, and acceptance rate is what
explains a difference from it.
