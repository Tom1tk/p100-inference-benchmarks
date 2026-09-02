# Methodology

Hardware, engines, models, and fixed parameters. Change nothing here without
recording why in [RUNLOG.md](RUNLOG.md). The objective and the current answer
are in [README.md](README.md).

---

## 1. Hardware

| | |
|---|---|
| Host | HP Z640, Proxmox, inference in an LXC. Xeon E5-2680 v4, 28 GB host RAM |
| GPUs | 2× Tesla P100-PCIE-16GB (16,269 MiB each), device 0 + 1 |
| Arch | GP100, compute capability 6.0 (sm_60) |
| Bandwidth | 732.2 GB/s HBM2 per card (the widely-quoted 549 GB/s is the 12 GB variant) |
| Interconnect | PCIe 3.0 PHB, **no NVLink**. The cards can peer both ways |
| Power | **175 W/card cap**, of a 250 W default. Approved ceiling 220 W (H15), gated on the user's PSU plug-meter check — see RUNBOOK §1. Persistence mode on. Neither survives a host reboot |
| Cooling | Passive cards, custom 3D-printed shroud + Arduino Nano PWM controller |
| Thermal limits | Idle ≤45 °C · load 70–80 °C · throttle 83 °C · hard shutdown 87 °C |

**sm_60 is the dominant architectural fact.** No dp4a, no tensor cores, no
async copy. Most modern quantised MMQ kernels assume sm_61+ and fall back to
slow generic paths — `GGML_CUDA_FORCE_MMQ` returns false unconditionally on
6.0, so quantised prefill always goes dequant → cuBLAS. GP100 does have a
native 2:1 FP16 rate (~18.7 TFLOPS/card), which cuBLAS already uses
(`CUBLAS_COMPUTE_16F` is selected for cc 600). Everything that needs sm_70+
(NCCL's internal pipeline, sparse attention kernels, PFlash, vLLM, SGLang,
TensorRT-LLM, ExLlamaV3) is out — see RUNBOOK §7 "Closed".

---

## 2. Engines

| Engine | Lineage | Commit | Binary directory |
|---|---|---|---|
| `mainline-rebased` | `ggml-org/llama.cpp` **PR #27342** (`dflash2`) replayed onto master `75844307` | `57affa09` | `/root/dflash2-rebased/build-cuda-p100/bin/` |
| `mainline` | PR #27342 as fetched | `64f765f5` | `/root/dflash2-llama.cpp/build-cuda-p100/bin/` |
| `buun` | `spiritbuun/buun-llama-cpp` (TurboQuant lineage) | `39d97a876` (build 11260) | `/root/buun-llama-cpp/build-cuda-p100/bin/` |
| `ik` | `ikawrakow/ik_llama.cpp` | `8337e4c` | `/root/ik_llama.cpp/build-nonccl/bin/` |
| `pflash` | `Tom1tk/mtp-pflash-turboquant-hip` | `e05ff58b7` (build 9110) | `/root/pflash-llama.cpp/build-cuda-p100/bin/` — **closed**, kept only for its `llama-niah` binary and fixtures |

`mainline-rebased` is the selected engine. `mainline` is a `git worktree` of
`/root/mainline-llama.cpp` on the fetched PR branch (`git fetch origin
pull/27342/head:pr-27342`). The PR was open and mergeable on 2026-08-24; if it
merges, pin the merge commit here. `scripts/build-rebased.sh` rebuilds the
rebase.

### Split-mode support, verified against each binary

| Engine | `none` | `layer` | `row` | `tensor` | `graph` |
|---|---|---|---|---|---|
| `mainline` / `rebased` / `buun` / `pflash` | ✅ | ✅ | ✅ | ✅ | ❌ |
| `ik` | ✅ | ✅ | ❌ | ❌ | ✅ |

`-sm tensor` is permitted for `qwen35` (`llm_arch_supports_sm_tensor()` is a
denylist) and it won Phase 1. **It requires `GGML_CUDA_ALLREDUCE=none` on this
rig**: the NCCL default aborts, and `internal` needs sm_70 and silently falls
back to the same butterfly path `none` uses.

### What can be combined

| Engine | `-sm tensor` | MTP | DFlash2 | TurboQuant KV |
|---|---|---|---|---|
| `mainline-rebased` | ✅ | ✅ | ✅ (layer only, H8) | ❌ |
| `buun` | ✅ | ✅ | ❌ | ✅ (`turbo*`, `vbr`) |
| `ik` | ❌ | ✅ | ❌ | ❌ |

⚠️ **`buun`'s default KV type is `vbr`, not `f16`.** Always pin `-ctk`/`-ctv`
on buun or the run is not comparable. The scripts do.

### Runtime knobs (mainline family, env vars, no rebuild)

| Env var | Note |
|---|---|
| `GGML_CUDA_ALLREDUCE` | `nccl` (default, aborts) / `internal` (falls back) / **`none`**. H5, H10 |
| `GGML_CUDA_P2P=1` | Peer access, off by default. +2.9% decode drafter-free, untested with MTP. H10 |
| `GGML_CUDA_CUBLAS_COMPUTE_TYPE` | `auto` already picks F16 on P100; `f32` is a diagnostic, not a lever |
| `GGML_CUDA_ENABLE_UNIFIED_MEMORY` | Oversubscribe VRAM at PCIe speed. Escape hatch, not a speed knob |
| `GGML_SCHED_DEBUG=1` | Prints the graph split assignment |

**Trap:** any `-ot` tensor override silently disables pipeline parallelism
(`src/llama-context.cpp` requires `LAYER` split, KV offload, and no overrides).
Compare `-ot` runs against a no-`-ot` control. `ik`'s graph-split knobs are
listed under H12 (parked).

### Flag syntax per engine

| | MTP | DFlash2 |
|---|---|---|
| `mainline` / `rebased` / `buun` | `--spec-type draft-mtp --spec-draft-n-max N -md <mtp.gguf>` | `--spec-type draft-dflash --spec-draft-n-max 7 -md <dflash2.gguf>` (`mainline` only) |
| `ik` | `--spec-type mtp:n_max=N,p_min=P` | — |
| `pflash` | `--spec-type mtp --spec-draft-n-max N` | — |

`draft-dflash` is the **same flag for v1 and v2**; the engine auto-detects v2
from the drafter's `dflash.selector_top_k` key. On `buun`/`ik` that line
silently runs v1. Confirm the selector loaded in the server log before
recording any DFlash2 number.

- `-fa 1` is portable across all engines; always pass it explicitly (`ik`
  defaults to 1, the others to 0/auto).
- `ik` accepts `-cuda graphs=0`.

### Build commands

```bash
# mainline family (dflash2-rebased, dflash2-llama.cpp). LLAMA_CURL=OFF: no
# libcurl headers here. buun/pflash: same without that flag.
cmake -B build-cuda-p100 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build-cuda-p100 --config Release -j12

# ik_llama — GGML_NCCL defaults ON and aborts on this rig
cmake -B build-nonccl -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 \
      -DCMAKE_BUILD_TYPE=Release -DGGML_NCCL=OFF
cmake --build build-nonccl --config Release -j12
```

Pascal is still first-class in mainline (`60 == P100, FP16 CUDA intrinsics` in
`ggml-cuda/CMakeLists.txt`), and PR #27342's one new kernel (`top-k.cu`) uses
nothing newer than sm_60.

---

## 3. Models

### The target is a hybrid, not a transformer

`Qwen3.8-27B` (`general.architecture = qwen35`, 27.32B params, 262k native
context) GGUF metadata:

```
qwen35.block_count             = 65     qwen35.full_attention_interval = 4
qwen35.ssm.state_size          = 128    qwen35.ssm.inner_size          = 6144
qwen35.ssm.group_count         = 16     qwen35.ssm.conv_kernel         = 4
qwen35.attention.head_count    = 24     qwen35.attention.head_count_kv = 4
qwen35.attention.key_length    = 256    qwen35.attention.value_length  = 256
```

Only ~16 of 65 layers are full attention; the other ~49 are gated-delta-net
(linear). Consequences:

- **KV is small**: 16 layers × 2 × 4 × 256 × 2 B = **64 KiB/token**, 6.25 GiB
  at 100k in f16, ~3.1 GiB at `q8_0`. 100k fits in VRAM either way.
- **Prefill is flat to 16k (−6%) but falls 34% by 100k** (H13). The GDN
  kernel walks tokens one at a time (`gated_delta_net.cu`, `//TODO: Add
  chunked kernel`) and is the short-context suspect (H18); the 16 attention
  layers' quadratic cost is the depth suspect (H23). The linear "2 × params"
  FLOP model that gives a 144 s hardware floor at 100k is **wrong at depth**
  because it ignores attention — do not quote %-of-peak figures for 100k.
- **Sparse attention has a low ceiling by layer count** but possibly not by
  FLOP count at 100k (H23). Moot on sm_60: every such kernel is sm_80+.

### Files on disk under `/root/`

| File | Size | Role |
|---|---|---|
| `Qwen3.8-27B-UD-Q4_K_M.gguf` | 15.3 GiB | **The serve quant** |
| `Qwen3.8-27B-UD-Q6_K_M.gguf` | 21.5 GiB | Phase 8 baseline candidate; marginal at 100k (~14.9 GiB/card predicted) |
| `Qwen3.8-27B-UD-Q5_K_M.gguf` | 18.4 GiB | Phase 8 |
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` | 16.4 GiB | Phase 8 / H2 only |
| `Qwen3.8-27B-UD-IQ3_S.gguf` | 11.2 GiB | The only quant that fits one card. Single-GPU is **closed** (H20/H25) |
| `Qwen3.8-27B-Uncensored-noMTP-Q4_K_M.gguf` | 15.4 GiB | H3 stock-quant test, confounded by fine-tune |
| `mtp-Qwen3.8-27B-Q4_0.gguf` | 1.28 GiB | MTP draft head (`nextn_predict_layers = 1`) |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` / `-Q8_0.gguf` | 1.06 / 1.92 GiB | DFlash2 drafters, v2 verified by `dflash.selector_*` metadata |

Verify an MTP head without loading the model:
`/root/benchmarks/benchvenv/bin/gguf-dump --no-tensors <model>.gguf | grep -i nextn`.

### Drafters are output-preserving

The target verifies every drafted token at temperature 0, so drafter choice
and drafter quant change **acceptance (speed) only**, never what is emitted.
None of the engines expose a lenient acceptance mode; `--spec-draft-p-min`
gates drafting, not acceptance. H24 confirmed it on this rig: 400 tokens
byte-identical across arms. This is why Phase 8's drafter sweep is a hash
comparison, not a quality sweep. DFlash drafters read the target's hidden
states, so a heavily quantised target can lower acceptance — still a speed
effect.

### Prior-work quant rules, unverified (H2, H3)

- Don't use `-XL` quants on a split: the Q8_0 embedding/output head can't be
  split and may stall one GPU.
- Stock quants below Q6 are claimed to degrade `ssm_out`; Unsloth Dynamic
  quants are claimed exempt.

---

## 4. Out of scope

**PFlash (product, fork, technique) and DFlash v1** are excluded. Both were
found too lossy in earlier work (DFlash v1 also mangled tool calls); PFlash
additionally needs sm_80 and has no v2. That "too lossy" verdict came from a
7900 XTX, so H17 does not confound it. Don't pass `--spec-type dflash` or
`draft-dspark` on `buun`/`pflash`.

**DFlash2 is in scope** (2026-08-24): separate checkpoint, separate PR, a
selector v1 lacks. It runs, but is layer-split-only (H8), so it lost to
tensor + MTP.

**In scope:** MTP, DFlash2, KV-cache quantisation (`q8_0` adopted; TurboQuant
measured and rejected).

---

## 5. Fixed parameters

**`llama-bench`** (Table A): each engine's own binary. No HTTP overhead, but it
cannot drive speculative decoding.

| Parameter | Value | Rationale |
|---|---|---|
| `-ngl` | `99` | Full offload |
| `-fa` | `1` | `-fa 0` aborts on sm_60 in graph-split mode |
| `-t` | `8` | GPU-bound; 6/8/10/12 within noise in prior work |
| `-ctk` / `-ctv` | `f16` | Bench default. The serve command uses `q8_0` (1.5% of decode, H25) |
| `-b` / `-ub` | `2048` / `2048` from Phase 7 on | Phases 1–6 ran at the default `-ub 512`, which H14 found is a local minimum: those prefill numbers are ~35% low but their rankings hold. **Always quote length and ubatch with a prefill figure** |
| `-r` | `3` | Mean ± stddev. Smoke tests used `-r 1` and are marked |
| `-p` | `0,2048,4096,8192,16384` (+ `65536,100000` for H13) | Prompt **lengths** into an empty cache (the cold-TTFT question). `-d` (depth) has never been used. `-p 0` yields no row |
| `-n` | `128` | tg128 |
| `CUDA_VISIBLE_DEVICES` | `0,1` | Re-confirm enumeration each session |

```bash
CUDA_VISIBLE_DEVICES=0,1 <engine-bin>/llama-bench -m <model>.gguf \
  -ngl 99 -fa 1 -t 8 -ctk f16 -ctv f16 -sm <split> -b 2048 -ub 2048 \
  -p 0,2048,4096,8192,16384 -n 128 -r 3 -o csv
```

**`llama-server`** (Table B): `scripts/run-spec-placement.sh`. 400 output
tokens, temperature 0, seed 42, `REPS=3` (2 at 100k), one slot. Decode t/s =
generated tokens ÷ wall time; **always record acceptance and depth alongside
it**. Two rules learned the hard way:

- **Never size a speculative run at 64 tokens** — it overstated MTP by 13×
  (90% acceptance vs 40% at 400 tokens).
- **Acceptance is not comparable across engines** unless `n_max`/`p_min` are
  pinned (buun 82.4% vs rebased 73.3% on the same drafter and prompt). Compare
  decode t/s instead.

`scripts/run-bench.sh` and `run-spec-placement.sh` wrap all of this plus
telemetry, logging and the commit/push. Prefer them over running by hand.
