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

### Split-mode support — verified, differs per engine

Checked directly against each binary's argument parser. The handover doc's
table was wrong about `ik`.

| Engine | `none` | `layer` | `row` | `tensor` | `graph` |
|---|---|---|---|---|---|
| `pflash` | ✅ | ✅ | ✅ | ✅ | ❌ |
| `buun` | ✅ | ✅ | ✅ | ✅ | ❌ |
| `ik` | ✅ | ✅ | ❌ | ❌ | ✅ |

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
```

The current `ik` build was made **without** `-DGGML_NCCL=OFF` and therefore
has NCCL linked in. This is the open blocker — see H5.

---

## 4. Models

Target: **Qwen3.8-27B** — 27.32B dense params, 262k native context,
architecture string `qwen35`, multimodal.

All files present on disk under `/root/`:

| File | Quantizer | Size | Role |
|---|---|---|---|
| `Qwen3.8-27B-UD-Q4_K_M.gguf` | Unsloth Dynamic | 15.33 GiB | Split-friendly, large KV budget |
| `Qwen3.8-27B-UD-Q5_K_M.gguf` | Unsloth Dynamic | 18.41 GiB | Midpoint |
| `Qwen3.8-27B-UD-Q6_K_M.gguf` | Unsloth Dynamic | 21.50 GiB | **Primary test quant** — handover doc's recommended split target |
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` | Unsloth Dynamic (XL) | 16.35 GiB | **H2 split-stall test only** — not a general split candidate |
| `Qwen3.8-27B-Uncensored-noMTP-Q4_K_M.gguf` | non-Unsloth ("stock") | 15.41 GiB | H3 stock-quant degradation test. Filename implies no MTP head — confirm before use |
| `mtp-Qwen3.8-27B-Q4_0.gguf` | — | 1.28 GiB | MTP draft head. Verified: `qwen35.nextn_predict_layers = 1`, arch matches base |

### Drafters (speculative decoding)

Three drafters are under test, plus a no-drafter control. All target
Qwen3.8-27B and pair with **any** quantisation of it — a drafter is trained
against a base model, not against a target quant.

| Drafter | Source | Engines that can drive it | Status |
|---|---|---|---|
| _(none)_ | — | all | Control |
| MTP head | `mtp-Qwen3.8-27B-Q4_0.gguf` (on disk, verified) | pflash, buun, ik | Ready |
| DFlash2 Q4_K_M | `z-lab/Qwen3.8-27B-DFlash2-GGUF` (~1.1 GB) | **none yet** — see H8 | Downloading |
| DFlash2 Q8_0 | `z-lab/Qwen3.8-27B-DFlash2-GGUF` (~2.0 GB) | **none yet** — see H8 | Downloading |

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
verdict does not transfer to it. No engine here can load it yet; see H8.

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
