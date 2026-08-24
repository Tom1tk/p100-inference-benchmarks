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
| Power | 175W/card cap, confirmed applied. Persistence mode enabled. Neither survives a host reboot |
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
| `mainline` | ✅ | ✅ | ✅ | ✅ | ❌ |

`-sm graph` exists **only** in `ik_llama`. `row`/`tensor` exist only in
`pflash`/`buun`. Passing an unsupported value gives
`error: invalid parameter for argument: -sm`.

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
| `-p` | `0,2048,4096,8192,16384` | Prefill depths to 16k. `-p 0` yields no output row — expected |
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

### MTP measurement (Phase 2)

Effective tg t/s = tokens generated ÷ wall time. **Also record the acceptance
rate** where the engine reports it — the comparison target is a community
reference of ~1.7× on Qwen3.6-27B on a 3090, and acceptance rate is what
explains a difference from it.
