# Run log

Chronological. Add an entry when something notable happens — a crash, a
parameter deviation, a rebuild, a surprising number. Routine successful runs
don't need narrative; they live in [RESULTS.md](RESULTS.md).

Newest last.

---

## 2026-08-20 — Setup

All three engines built for sm_60 and confirmed working (`--version` succeeds,
both P100s enumerate, `llama-bench` present):

- `pflash` @ `e05ff58b7` (build 9110)
- `buun` @ `39d97a876` (build 11260)
- `ik` @ `8337e4c` (build 4430-era lineage, "Fix Qwen35+ MTP")

Model set complete — all six files on disk, sizes recorded in
[METHODOLOGY.md](METHODOLOGY.md) §4. MTP draft head verified via `gguf-dump`
(`qwen35.nextn_predict_layers = 1`, architecture matches base model).

---

## 2026-08-20 — Phase 0 smoke test #1: pflash, layer split

`phase0-pflash-layer-q6k` — pflash, `UD-Q6_K_M`, `-sm layer`, full prefill
depth list, **`-r 1`**. Single round deliberately: this was a thermal
validation run to check cooling before committing to the full matrix, not a
Phase 1 data point.

**Result: clean.** 187.99 / 197.04 / 198.06 / 193.46 t/s prefill at
2k/4k/8k/16k, 9.46 t/s tg128.

**Cooling: fine.** Peak 63°C on both cards against an 83°C throttle. Both GPUs
alternated to 100% utilization as expected for layer split — the dual-GPU path
is genuinely engaging both cards.

Two observations worth recording:

- Model load was much slower than expected (~7 min) because the two remaining
  model downloads were saturating disk I/O concurrently. Affects wall-clock
  only, not GPU-phase numbers. Both downloads have since completed.
- Power draw briefly read ~180W against the 175W cap. Investigated: the cap
  **is** correctly applied (`power.limit = 175.00 W`, persistence enabled on
  both cards). Instantaneous sample overshoot above a rolling-average cap is
  normal. Not an issue — noted here because the earlier draft of this log
  flagged it as an open question, and it isn't one.

---

## 2026-08-20 — Phase 0 smoke test #2: ik_llama, graph split — FAILED

`phase0-ik-graph-q6k` — ik, `UD-Q6_K_M`, `-sm graph -cuda graphs=0`, otherwise
identical parameters.

**Result: crashed** before producing any throughput data.

```
ggml_cuda_op_reduce: ncclAllReduce failed with status 1
=============================== NCCL main communicator initialized
/root/ik_llama.cpp/ggml/src/ggml-cuda/reduce.cu:169: Fatal error
```

Abort on the first prefill test, via
`ggml_cuda_op_reduce` → `ggml_cuda_compute_forward` →
`ggml_backend_cuda_graph_compute`. Not thermal — GPUs never left idle (peak
50°C).

**Diagnosis:** `GGML_NCCL` defaults to **ON** in ik_llama's CMake
(`ggml/CMakeLists.txt:97`), and the current build was made without overriding
it. The handover doc warned "do not build with NCCL" for this rig on
performance grounds (no NVLink, so AllReduce over PCIe loses to direct P2P) —
the reality on this build is stronger: it doesn't work at all.

**Fix (known, not yet applied):** rebuild with `-DGGML_NCCL=OFF`.

**Impact:** blocks `-sm graph` entirely, which blocks H6. `none`/`layer` split
modes on ik are untested but shouldn't invoke the NCCL reduce path. See H5 in
[HYPOTHESES.md](HYPOTHESES.md).

---

## 2026-08-20 — Documentation restructure

Split the single `BENCHMARK_PLAN.md` into this repo's document set (README /
RUNBOOK / METHODOLOGY / HYPOTHESES / RESULTS / RUNLOG) and initialized git with
a remote, so results are versioned and pushed per run rather than accumulating
in one growing local file.

Two errors in the original plan document were found and corrected while
verifying it against the actual binaries:

1. **`ik_llama`'s split-mode support was wrong.** The plan claimed
   `none, layer, row, tensor, graph`. Verified against the argument parser:
   `ik` accepts only `none|layer|graph` — `row` and `tensor` are rejected with
   `error: invalid parameter for argument: -sm`. An agent following the old
   table would have hit avoidable failures.
2. **`buun`'s `-fa` value format** is documented in its help as
   `<on|off|auto>` rather than `<0|1>`. Verified that it accepts `1` as well,
   so `-fa 1` remains portable across all three engines — but the discrepancy
   is now recorded rather than left to be rediscovered.

Also confirmed and recorded: power cap and persistence mode are correctly
applied; both outstanding model downloads have completed.

## 2026-08-20 — Phase 1 cell: pflash, layer split

`phase1-pflash-layer-q6k` — pflash, `UD-Q6_K_M`, `-sm layer`, full prefill
depth list, **`-r 3`**. First Phase 1 baseline cell.

**Result: clean.** pp2048 188.46 / pp4096 197.11 / pp8192 198.21 / pp16384
193.38 t/s, tg128 9.45 t/s. Peak 65°C (within the 63–65°C range observed in
Phase 0; throttle 83°C). Both GPUs alternated to 100% in the layer split.

Batch continued automatically, but was stopped after this cell per operator
direction (benchmark, log, commit, stop).

---

## 2026-08-20 — Phase 1 cell: pflash, none split — BLOCKED

`phase1-pflash-none-q6k` — pflash, `UD-Q6_K_M`, `-sm none` (single-GPU
reference). **Failed to load model** within ~6s:

```
main: error: failed to load model '/root/Qwen3.8-27B-UD-Q6_K_M.gguf'
```

**Diagnosis:** `-sm none` is the single-GPU reference in METHODOLOGY §3.
`-ngl 99` places the whole 21.49 GiB model on one 16 GB card, which cannot
hold it (weights + f16 KV). This is a hardware limit, not a transient error —
the model loaded fine immediately after in `layer` split. The `none` reference
cell is therefore **not achievable** for Q6_K_M on this rig and is recorded as
blocked rather than re-run.

Note: this run occurred while the disk was at 100% usage (before a +25GB
resize), but the model file was intact and loaded successfully moments later
in layer mode, so the disk state was not the cause.

---

## 2026-08-24 — Phase 5 added: agentic web-build benchmark

Added a final phase that measures **usability** rather than throughput: `pi`
drives each model version through a three-stage one-shot task (build an Express
site about itself, restyle it with a working JS interactive, add a canvas
mini-game in a separate static file). Procedure in
[WEB_BENCH.md](WEB_BENCH.md).

Per-request prefill/decode t/s are captured by a recording proxy
(`scripts/web_bench_metrics.py`) between the agent and `llama-server`. A proxy
rather than log-scraping because all three engines emit the same `timings`
object in their OpenAI-compatible responses but their log formats differ —
verified in `tools/server/server-task.cpp` (pflash, buun) and
`examples/server/server-task.cpp` (ik).

**Port scheme changed from the original prompt.** The prompt hardcoded
`localhost:4000`, so only one site could be hosted at a time. Every port now
derives from a per-run index: site `4000+i`, server `8100+i`, proxy `8200+i`.
Sites are deliberately left running, so at the end of the phase every model
version is browsable side by side. The prompt also now pins the systemd unit
name to the run label, which the original didn't — without that the units
collide even with distinct ports.

Verified before wiring anything up, against a mock OpenAI server:

- `pi` resolves a local provider from `models.json` with
  `api: "openai-completions"`, and uses **streaming** at
  `/v1/chat/completions`.
- **`PI_OFFLINE=1` is required.** Without it `pi` blocks on startup network
  operations in this environment and never issues a request — it looks exactly
  like a model hang. The script sets it; recorded in RUNBOOK §5.
- The proxy records correct per-stage timings through both the streaming and
  non-streaming paths, and `summarize` reduces them to avg/min/max.

The agent's `pi` config is written per-run under `sites/<label>/.pi-agent/`;
the operator's `~/.pi/agent` is never modified.

---

## 2026-08-24 — DFlash2 re-opened, and blocked

The operator asked to revisit DFlash2 after previously writing off DFlash as
too lossy and prone to mangling tool calls. **The exclusion is correctly
re-opened** — DFlash2 is a different drafter, not a newer build of the same one:
separate checkpoint, separate upstream PR (#27342 vs #22105), and a
candidate-path selector v1 doesn't have. The PFlash / DFlash-v1 exclusion still
stands. METHODOLOGY §5 amended accordingly.

**It cannot run here yet.** Engine survey:

| Engine | `--spec-type` offers | DFlash2? |
|---|---|---|
| `pflash` | `mtp`, `ngram-*` only | No DFlash at all |
| `buun` | `draft-dflash`, `draft-dspark`, `dflash` | v1 — docs cite #22105 |
| `ik` | `dflash`, `dspark` | v1, same lineage |

No commit in `buun` references #27342, and `common/speculative.cpp` has no
candidate-path selector. Risk noted in H8: the v1 loader may *accept* a v2 GGUF
and silently draft without the selector, which would present as poor acceptance
rather than a clean error. A v1-loader number must not be recorded as a DFlash2
result.

Second concern, from `buun`'s own `CLAUDE.md` and `src/llama-context.cpp:3343`:
DFlash tree verify (`parent_ids_gpu`) is GPU-0 only and **auto-disabled when
`n_devices() > 1`**. On this dual-GPU rig that path is off by construction. If
DFlash2's selector depends on it, DFlash2 may be single-GPU-only here.

Also established while answering a drafter-pairing question, and worth keeping:
**speculative decoding on these engines is output-preserving.** The target
verifies every drafted token over its full vocabulary, and none of the three
engines exposes a lenient-acceptance flag (`--spec-draft-p-min` gates drafting,
not acceptance). So drafter choice and drafter quantisation are speed/VRAM
variables, not quality ones — which makes H9 (Q4 vs Q8 drafter) an acceptance
rate comparison. Recorded in METHODOLOGY §4.

Drafter arms for Phase 5 are now: none (control), MTP, DFlash2-Q4, DFlash2-Q8.
The first two are runnable today; the DFlash2 arms are gated on H8.

---

## 2026-08-24 — DFlash2 unblocked: mainline PR #27342 added as a fourth engine

The previous entry recorded DFlash2 as blocked because none of the three forks
supports it. That was right about the forks and wrong to stop there — the
operator pushed back, correctly. Upstream **`ggml-org/llama.cpp` PR #27342**
("spec : add DFlash2 support (local convolution + candidate selector)") is the
engine, and it builds for Pascal.

PR state when checked (2026-08-24): **open**, `mergeable: MERGEABLE`, base
`master`, branch `dflash2`, 16 files, +563/−7, last updated the same day. Not
merged, so this is a PR build and must be re-checked before any `mainline`
number is called reproducible from a release.

Set up as a worktree so `master` stays clean and the PR can be refreshed:

```bash
cd /root/mainline-llama.cpp
git fetch origin pull/27342/head:pr-27342
git worktree add /root/dflash2-llama.cpp pr-27342     # 64f765f5
cmake -B build-cuda-p100 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
```

`LLAMA_CURL=OFF` because there are no libcurl dev headers here and every model
is local. OpenSSL is likewise absent — cpp-httplib warns and drops HTTPS, which
is irrelevant for a loopback server.

**Pascal risk assessed before building, not after.** mainline still documents
`60 == P100, FP16 CUDA intrinsics` in `ggml/src/ggml-cuda/CMakeLists.txt`, and
CUDA 12.0 takes the `VERSION_LESS "13"` branch. The PR's only new CUDA kernel,
`ggml/src/ggml-cuda/top-k.cu` (a two-stage tiled top-k), uses shared memory and
`__syncthreads()` and nothing else — no sm_70 warp reductions, no tensor cores,
no async copy, no bf16. It compiled clean for `sm_60`.

### The v1-silently-runs risk is now confirmed, not hypothetical

PR #27342 adds **no new `--spec-type` value**. The same `draft-dflash` runs v1 or
v2, decided by draft metadata in `common/speculative.cpp`:

```cpp
selector_top_k = llama_model_dflash_selector_top_k(model_dft);
is_dflash2     = selector_top_k > 0;
```

So a `buun`/`ik` run with a v2 drafter takes the v1 path on an identical command
line and reports plausible-looking numbers. Recorded as a failure mode in
RUNBOOK §5: confirm the selector loaded before recording anything as DFlash2.

Both drafters were verified as genuine v2 by metadata rather than filename:
`dflash.selector_rank = 256`, `dflash.selector_top_k = 16`, plus
`selector_hidden` / `selector_predecessor` / `selector_successor` tensors — the
exact tensors the PR adds mappings for. Actual sizes 1.06 GiB (Q4_K_M) and
1.92 GiB (Q8_0), smaller than the ~1.1/2.0 GB first assumed.

### The dual-GPU concern does not apply here

Flagged earlier as a possible blocker on the strength of `buun`'s
`n_devices() > 1` guard. That guard is `buun`-specific — it disables
`parent_ids_gpu` **tree** verification. Mainline has no tree-verify path at all
(`parent_ids` appears nowhere in `src/` or `common/`), and the PR's selector
emits a **linear** draft: it walks a predecessor chain and `push_back`s a flat
sequence, verified the ordinary way with no device restriction. The only
`n_devices() > 1` in mainline's context is the unrelated pipeline-parallelism
check.

Conclusion: DFlash2 should run on `-sm layer` across both cards. This is a code
reading and still needs an empirical load check — H8 step 1.

## 2026-08-24 — `UD-IQ3_S` added to the quant sweep

Operator downloaded `Qwen3.8-27B-UD-IQ3_S.gguf` (~11 GiB) to make single-GPU
DFlash2 testing possible if the dual-GPU path misbehaves.

It earns a permanent place in Phase 3 for a second reason: **it is the only
target that fits one 16 GiB P100 with a usable KV cache.** The Phase 1 `none`
leg is currently recorded as blocked — "21.5 GB model doesn't fit one 16 GB
card, `failed to load model`" — which also blocks the single-vs-dual comparison
H6 needs. `IQ3_S` unblocks it. Phase 3 now runs it twice: `-sm layer` for the
quant-curve row, `-sm none` for the single-GPU reference.

Caveat recorded in METHODOLOGY: expect its prefill to trail the K-quants. IQ
dequantization is compute-heavy and sm_60 has no dp4a, so its numbers measure
the cost of single-GPU operation, not a like-for-like quant comparison.

**Disk pressure.** `/` hit 94% during this work (11 GiB free) with five 15–22 GiB
targets, the IQ3_S download, and the new build all resident. Not yet a failure,
but the next large download needs space cleared first.

---

## 2026-08-24 — DFlash2 confirmed working on dual P100s

`h8-loadcheck-df2q4`. First DFlash2 inference on this rig, and the answer to the
dual-GPU question that was flagged as a possible blocker.

Command:

```bash
CUDA_VISIBLE_DEVICES=0,1 /root/dflash2-llama.cpp/build-cuda-p100/bin/llama-server \
  -m /root/Qwen3.8-27B-UD-Q6_K_M.gguf \
  -md /root/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  -ngl 99 -ngld 99 -fa 1 -t 8 -c 4096 -sm layer --jinja
```

Result: loaded in 4m40s, **13.8 GiB on GPU0 and 15.0 GiB on GPU1**, decode
**16.10 t/s** at **57.0% acceptance** (239 of 419 drafted), output correct and
unmangled. Peak 49°C.

**Dual-GPU works.** The `buun` `n_devices() > 1` tree-verify restriction does not
apply to mainline, exactly as the code reading predicted. The single-GPU `IQ3_S`
path remains valuable as a quant row and the H6 single-GPU reference, but is not
needed to rescue DFlash2.

**Do not quote 16.10 t/s as "the" DFlash2 number.** One prompt, 300 tokens,
greedy, 4k context, via the server rather than `llama-bench`. Comparing it to the
Phase 1 tg128 of 9.45 t/s suggests ~1.7× but confounds drafter with engine, tool,
and context depth — the exact mistake this repo exists to avoid. mainline's own
no-drafter control is the next run.

### Two useful findings

**Acceptance rate is free.** The response `timings` object carries `draft_n` and
`draft_n_accepted`. `scripts/web_bench_metrics.py` already records the full
timings object, so Phase 5 gets per-request acceptance with no code change and no
log parsing — and it sidesteps the v1/v2 verification problem entirely, since the
startup log prints the shared dflash params but not the branch taken.

**Benign-looking startup lines that are not failures**, now that a successful run
is on record to compare against:

- `dflash requires ctx_other to be set (this warning is normal during memory
  fitting)` — self-labelled, appears on a healthy load.
- `[spec] failed to measure draft model memory` — follow-on from the above.
- `common_fit_params: failed to fit params ... n_gpu_layers already set by user
  to 99, abort` — auto-fit declining because `-ngl` was pinned. Note the
  consequence: **mainline will not shed layers to make a too-large config fit**;
  a genuine VRAM overflow fails hard.
- `model has unused tensor blk.64.*` — the target's own MTP head being ignored
  because an external drafter was supplied. Correct, and a useful confirmation
  that DFlash2 rather than `nextn` is driving speculation.

### Process note

A health poll of `curl -s .../health` reported success against an HTTP 503
`{"error":"Loading model"}` and produced a false "server up". `run-web-bench.sh`
was checked and is **not** affected — it uses `curl -sf`, where `-f` makes 5xx
non-zero. Recorded because the failure mode is silent: the ad-hoc check looked
like it had worked.

**Disk:** operator added 25 GB mid-session; `/` now 197 GB total, 30 GB free
(85%). The earlier 97% pressure is resolved.

---

## 2026-08-24 — Dual-GPU capability survey: what we haven't been testing

Prompted by "cover everything available to us." Surveyed all four engines'
`--help` and source rather than trusting the existing table. Three corrections
and one new phase.

### Correction: `-sm tensor` is unusable for this model, on every engine

The split-mode table listed `tensor` as supported on `pflash`/`buun`/`mainline`
because the argument parser accepts it. It is gated per-architecture, and
`qwen35` is not on the list in **any** of the three trees — checked
`llm_arch_supports_sm_tensor()` in each. Mainline lists 30 architectures (Grok,
MPT, Deepseek2/32/4, Mamba/Mamba2, Jamba, Nemotron-H, MiniMax, LFM2, …); the
only Qwen entry is `QWEN3TTS`, a different architecture. It throws at load:

```
LLAMA_SPLIT_MODE_TENSOR not implemented for architecture 'qwen35'
```

Table now marks it ⚠️ with the explanation. **The real matrix for this model is
`none` / `layer` / `row`, plus `graph` on `ik` only** — four cells fewer than
previously implied. `-sm graph` remains `ik`-exclusive; `mainline` does not have
it.

### Finding: peer access is off by default in mainline

`cudaDeviceEnablePeerAccess` is called only when `GGML_CUDA_P2P` is set
(`ggml/src/ggml-cuda/ggml-cuda.cu`). Probed the hardware directly with
`cudaDeviceCanAccessPeer`: **1 in both directions**, PHB topology, single NUMA
node. So the cards can talk directly and, by default, don't. Now H10 — and it
applies to the config that already works, which makes it the cheapest
outstanding test in the project.

### Finding: mainline's AllReduce backend is runtime-switchable

```cpp
const char * env = getenv("GGML_CUDA_ALLREDUCE");   // "nccl" | "internal" | "none"
if (!env) { /* Linux default */ ggml_backend_cuda_comm_init_nccl(ret); }
```

`internal` is a dedicated **two-device** pipeline — it warns
`internal AllReduce init failed (n_devices != 2?)` on fallback, so the fast path
is written for exactly this rig's shape.

This is H5's other half, and it changes that hypothesis's practical advice.
`GGML_CUDA_NCCL:BOOL=ON` was auto-detected in our mainline build, so NCCL *is*
linked and *is* the Linux default — the same configuration that aborts on `ik`.
`h8-loadcheck-df2q4` passed only because `-sm layer` never invokes AllReduce.
**A `-sm row` run on mainline may reproduce `ik`'s abort** — worth knowing before
it looks like a new bug. Unlike `ik`, the fix needs no rebuild: set the env var.

### Trap worth recording: `-ot` silently disables pipeline parallelism

```cpp
bool pipeline_parallel =
    model.n_devices() > 1 && model.n_gpu_layers() > model.hparams.n_layer_all &&
    model.split_mode() == LLAMA_SPLIT_MODE_LAYER && cparams.offload_kqv &&
    !model.has_tensor_overrides();
```

Pipeline parallelism is off under `-sm none`, `-sm row`, `--no-kv-offload`, or
any `-ot`. Since `-ot` is the obvious tool for H2's XL split-stall problem, an
`-ot` experiment that appears to help or hurt may just be measuring the loss of
pipeline parallelism. Any `-ot` run needs a no-`-ot` control.

### Also catalogued (untested)

`ik` ships graph-split tuning that has never been exercised: `-smf16`/`-smf32`/
`-grt` (inter-GPU exchange precision — plausibly meaningful given 2:1 FP16 and
no NVLink), `-gap` (FA precision under graph split), `-smgs`, `-sas` (async
graph eval). All blocked behind the H5 rebuild. Now H12.

Draft-model placement (`-devd`, `-otd`) exists on all three llama.cpp-descended
engines and is untested — now H11, and it interacts with H9 since the Q8 drafter
is 0.86 GiB larger than the Q4.

`GGML_CUDA_CUBLAS_COMPUTE_TYPE` was checked and is **not** an opportunity: auto
already selects F16 for quantized mat-muls on fast-FP16 hardware, which P100 is.
`GGML_CUDA_PDL` is Hopper-era and irrelevant on sm_60.
`GGML_SCHED_DEBUG=1` prints graph split assignment — adopted as the diagnostic
of record for proving H2 numerically.

Added as **Phase 6** in RUNBOOK §7, ordered cheapest-and-most-applicable first.

---

## 2026-08-24 — H11 (drafter placement) run: refuted, plus an upstream abort

10 arms on `mainline`, target `UD-Q4_K_M`, `-sm layer`, `-c 4096`, 400 tokens,
temp 0, seed 42, 3 reps: {MTP-Q4_0, DFlash2-Q4_K_M, DFlash2-Q8_0} x {default,
`-devd CUDA0`, `-devd CUDA1`} plus a no-drafter control. 8 arms produced data,
2 aborted. New harness: `scripts/run-spec-placement.sh` + `scripts/run-h11-matrix.sh`.

**H11 is refuted.** Decode spread within a drafter is <=0.15% — MTP
14.11/14.12/14.11 t/s across its three placements, DFlash2-Q4 14.46/14.48.
Acceptance was identical across placements of the same drafter to the exact
draft-token count (MTP 228/513 in all three), which is what makes this a real
null and not a noisy one: had anything but locality moved, acceptance would have
moved with it. VRAM did shift between cards (GPU0 8647-10291 MiB), so the flag
demonstrably works — decode just does not care. `-devd`/`-otd` come out of the
Phase 5 matrix.

I had predicted `-devd CUDA1` would win, on the reasoning that the last target
layers and sampling live there under `-sm layer`. It did not; nothing did.

### `-devd CUDA0` aborts with any DFlash2 drafter

Reproducible on both quants, at load, before a single token:

```
llama_init_from_model: failed to initialize the context: dflash requires ctx_other to be set
srv load_model: [spec] failed to measure draft model memory
ggml-backend.cpp:930: pre-allocated tensor (output.weight) in a buffer (CUDA1)
                      that cannot run the operation (NONE)   -> ggml_abort
```

The DFlash2 drafter GGUF ships **neither `output.weight` nor
`token_embd.weight`** (checked with `gguf.GGUFReader`; it has 81 tensors and
only `output_norm.weight` of that group) and borrows the target's through
`cparams.ctx_other`. Under `-sm layer` the target's `output.weight` is on the
last device, CUDA1. `-devd CUDA0` restricts the draft context to CUDA0, the
borrowed tensor is then in a buffer the draft scheduler may not use, ggml aborts.

That single fact predicts the whole matrix: MTP carries its own copies of both
tensors, so all three of its placements load; DFlash2 survives `default` (both
devices permitted) and `CUDA1` (where the tensor already is) and fails only on
`CUDA0`. Not a VRAM limit — GPU0 had ~6 GiB free. Upstream bug in PR #27342.

### First controlled speculative-decoding speedups

The no-drafter control on the same engine/tool/context is **12.79 t/s**, so:
MTP **+10.3%**, DFlash2 **+13.0%**. The earlier uncontrolled 9.45 -> 16.10 t/s
pairing implied ~70%; it differed in engine, tool *and* context depth and should
not be quoted. Also worth noting acceptance does not translate proportionally —
DFlash2 accepts 45.8% of drafted tokens and returns 13%, because the verify pass
is not cheap on P100.

### H9 evidence, unplanned

DFlash2-Q8_0 accepted **228/513 = 44.4%**; DFlash2-Q4_K_M accepted
**231/504 = 45.8%**, ran fractionally faster (14.48 vs 14.46 t/s) and used
**0.87 GiB less VRAM**. One prompt at temp 0 — exact, but not general. The Q8
drafter currently has no measured advantage to justify its footprint.

### Process notes

- **`prefill_tps` in this CSV is not usable.** `cache_prompt: false` did not
  defeat the server's slot-level LCP prefix reuse (`f_sim_best = 1.000` in the
  logs), so reps 2-3 skip most of prefill — rep 1 read 22.40 t/s and rep 2
  43.72. Decode and acceptance are unaffected. Any future prefill measurement
  needs a distinct prompt per rep, not a cache flag.
- **Rep 1 is a cold-cache outlier** and is excluded from the means; the reported
  figure is the rep 2-3 mean. First load took 182s at ~112 MB/s disk-bound;
  subsequent loads of the same target hit page cache and took 10-35s.
- Peak temp across all 10 arms was 60°C, 23°C under the 83°C limit.

---

## 2026-08-24 — H11 depth check, and H11 closed

`-c 4096` with a ~50-token prompt only ever decodes at ~450 tokens of real
depth — `-c` sizes the KV allocation, not the occupancy. Since placement is a
locality claim and inter-GPU traffic scales with depth, the 4096 null was not
sufficient on its own. Re-ran `default` vs `-devd CUDA1` (DFlash2-Q4) at
`-c 16384` behind a 14.5k-token prompt built from the repo's own docs — real,
varied, deterministic text rather than repeated filler.

| Placement | Decode | Prefill | Acceptance | VRAM 0/1 | GPU1 free |
|---|---|---|---|---|---|
| `default` | 17.10 t/s | 160.0 t/s | 263/406 = 64.8% | 10201/11087 | 5182 MiB |
| `-devd CUDA1` | 17.06 t/s | 160.1 t/s | 263/406 = 64.8% | 9079/11887 | 4382 MiB |

0.2% apart, `default` fractionally ahead, acceptance identical to the token.
**H11 is refuted at both depths and is now closed.**

Settled the standing recommendation: **do not pass `-devd`.** `CUDA1` was
tempting because it is the one pin DFlash2 tolerates, but `default` tolerates it
too — only `CUDA0` aborts — so there is no compatibility argument, and pinning
costs 800 MiB of headroom on the already-fuller card at 16k, growing with
context.

Also retired the load-time argument for pinning. The apparent "CUDA1 loads
faster" pattern (10s vs 21s/35s) was page-cache first-touch ordered by run
sequence: every `default` arm happened to be the first read of its drafter file,
every `CUDA1` arm ran third in its group. The controlled pair is
`h11-mtp-cuda0` and `h11-mtp-cuda1` — same drafter, same cache state — at 10s
and 10s. Run 2 also loads *no* drafter in 20s, slower than run 4 loads one in
10s. Failed arms' 20s/40s are abort-plus-backtrace, not load.

### Incidental: decode is faster at depth, not slower

17.10 t/s at 14.5k vs 14.46 t/s at 450. Acceptance rose 45.8% -> 64.8%:
summarising a supplied document is much more predictable than free-form
generation, and that outweighed the deeper-KV cost. A caution against reading
any single-prompt acceptance number as a property of the drafter — it is a
property of drafter *and* task.

### Harness bugs found and fixed

- **First attempt at this run failed all six requests and recorded no reason.**
  `curl -sf` hides the response body, so the harness logged three
  `FAILED(request)` rows per arm with nothing to diagnose. The server had said
  `request (18313 tokens) exceeds the available context size (16384 tokens)` —
  my chars-per-token estimate was 4.0 when the real ratio is 3.18. Switched to
  `curl -s` plus an explicit `"timings"` check that prints the server's error
  into the log. The bogus rows were removed from the CSV; they were a harness
  defect, not a measurement.
- `scripts/run-spec-placement.sh` gained `CTX` and `PROMPT_FILE` env overrides.

---

## 2026-08-25 — Phase 1 engine baseline on `UD-Q4_K_M`

Six cells, `scripts/run-phase1.sh`, run detached. Five completed, one aborted
by design (`ik` with NCCL — that abort *is* the H5 result). Chose `UD-Q4_K_M`
over Q6 for speed; the whole phase took ~64 min of GPU time and never exceeded
68°C.

Deliberately **not** run: `ik` with NCCL on `layer`. We only want `ik` for
`-sm graph`, so the NCCL question only needs answering on the mode we'd
actually use.

### Two conclusions were overturned mid-phase

Worth recording, because in both cases the first result was the misleading one.

**1. "graph beats layer by +91%" — measured against a broken baseline.**
The first two cells were ik graph vs ik layer, and graph won both axes.
I flagged the suspicious part at the time: the odd number wasn't graph's 153
at pp16384, it was *layer's 80*. Three subsequent engines put layer at
187.87 / 198.67 / 188.69 on identical work. `ik`'s layer implementation is the
outlier, 2.3× off the field, and comparing graph to it inflated graph's case.
Against a competent layer implementation graph is a **trade**: +76% decode,
−23% deep prefill, crossover near 6k.

This also retires an earlier speculation of mine *and* its retraction. I first
guessed graph trades prefill for decode; then withdrew that when the
same-binary comparison showed graph winning both. The original guess was right
at the cross-engine level and wrong within `ik` — the within-engine comparison
simply had a defective control.

**2. `-sm tensor` is not arch-blocked for `qwen35`. I had the gate backwards.**
Recorded previously as unusable because `qwen35` is absent from
`llm_arch_supports_sm_tensor()`. It is a **denylist**:

```cpp
case LLM_ARCH_QWEN3TTS:
    return false;
default:
    return true;      // qwen35 lands here
```

Absence means *permitted*. True on our built branch too, not just upstream
tip — `build-cuda-p100/bin/llama-bench` advertises `<none|layer|row|tensor>`.
The decisive cell for the whole ik-lock-in question needs no rebuild and has
not been run.

### Upstream drift check

`origin/master` (`75844307`, Aug 24) is **115 commits** ahead of our PR
branch's base. Reviewed all of them for relevance to this rig:

- **Worth having:** `84979813` (backend split scheduler race — splits without
  input running concurrently while reusing another split's memory; we run two
  GPUs on every cell), `d9b6be07` (cuBLAS static workspace, 4 MiB/stream —
  cuBLAS FP16 GEMM is our entire prefill path), `2c6b141e` + `f466cfa3`
  (speculative/MTP fixes, bear on Phase 2).
- **Inert for us:** `2b562109` "CUDA: switch points per HW and quant type to
  tune the mvq->MMQ decode crossover" sounds targeted but its tables cover only
  Ada/Blackwell/DGX Spark/CDNA, and MMQ is unreachable on our build anyway —
  `ggml_cuda_should_use_mmq` returns false when
  `highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A` (610) and we compile for 60.
  The new `GGML_CUDA_MMVQ_MAX` env knob is therefore a no-op here; don't chase
  it. `d59d455f` (tensor-split meta backend fixes) was reverted by `f20395da`.
  Zero Pascal/sm_60-specific changes in the entire `ggml-cuda` diff.
- The "large release" itself is `bb4caa75`, a version bump to 0.2.0 plus
  release tooling — process, not code.

The DFlash2 PR merges onto current master **conflict-free**
(`git merge-tree --write-tree origin/master pr-27342` → exit 0, tree OID only,
no conflict paths). So rebasing costs nothing but a rebuild; the PR branch does
not have to be given up to pick up the fixes above. Not done yet — it would
invalidate the comparability of the `mainline` row above.

### Harness/data bug found

`results/all-results.csv` concatenates three different llama-bench schemas
(pflash 43 columns, buun 46, `ik` 56) under whichever header was written first.
`avg_ts` therefore sits at a different index per engine, and both positional
and `DictReader` access return the wrong column silently — it read `ik`'s
decode as 0.02 t/s instead of 22.05. Parse `results/raw/<label>.csv` instead;
each carries its own correct header. Documented in RUNBOOK.md §4.

## 2026-08-25 — `-sm tensor` on mainline, and the DFlash2 rebase

Two items closed. The single-GPU (`none`/IQ3_S) leg stays deferred.

### `-sm tensor` wins, and `ik` lock-in was never real

Three cells: mainline `-sm tensor` under each AllReduce backend.

The NCCL leg aborted in 5s. The harness log truncated the message to
`ggml-cuda.cu:106: CUDA error`, which is just the generic abort dispatcher, so
I reproduced it by hand to get the backtrace — worth doing rather than guessing:

```
#3 ggml_cuda_error(...)
#4 ggml_backend_cuda_comm_allreduce_nccl(...)
#5 ggml_backend_meta_graph_compute(...)
```

That is H5 again, in a second engine. NCCL AllReduce is broken rig-wide, not in
`ik` specifically — I had recorded the opposite in H5 and have corrected it.
Layer split was never affected because it performs no AllReduce, which is why
every `-sm layer` cell ran fine on NCCL-linked builds.

Mainline exposes a runtime escape (`GGML_CUDA_ALLREDUCE`) where `ik` needs a
rebuild. With `internal`:

| test | mainline tensor | ik graph | mainline layer |
|---|---|---|---|
| pp2048 | **214.83** | 199.14 | 180.38 |
| pp4096 | **219.13** | 198.45 | 190.96 |
| pp8192 | **212.70** | 179.56 | 192.14 |
| pp16384 | **206.32** | 153.04 | 188.69 |
| tg128 | 20.34 | **22.05** | 12.88 |

Fastest prefill in the entire matrix at every depth, +35% over graph at 16k,
holding shape (−6% from its 4k peak vs graph's −23%), for 7.8% of graph's
decode. Both cards at 97–99% utilisation and 170–180W throughout — real
parallel work, unlike layer split where one card idles.

`internal` vs `none` came out identical: every cell within 0.2%, most within
0.05%, stddevs 0.02–0.16, both runs 608s to the second. No
`internal AllReduce init failed` warning, so the dedicated two-device pipeline
did initialise and still changed nothing. AllReduce is a correctness switch on
this rig, not a performance one. That is half of H10 answered; `GGML_CUDA_P2P`
is the remaining half and is now the best cheap test available, since tensor
split moves far more cross-GPU data than the `-sm layer` config H10 originally
proposed testing on.

**Third correction to the `-sm tensor` story.** I had it wrong twice: first that
the arch gate blocked `qwen35` (it is a denylist — absence means permitted),
then, when it did fail, the cause was NCCL rather than the gate. The gate was
never involved.

### Rebase

`pr-27342-rebased` at `/root/dflash2-rebased` (separate worktree, so the
original branch and its binary stay intact and the Phase 1 rows stay
reproducible). All 12 commits replayed onto `75844307` with zero conflicts, as
the `merge-tree` dry-run predicted. Verified `84979813`, `d9b6be07`, `2c6b141e`
and `f466cfa3` are all ancestors and DFlash2 content survived.

The rebase also absorbed two upstream edits to `dflash.cpp` itself: the
`ggml_rope_set_offset` refactor replacing a manual view/rope/concat, and
`output` tensor creation moved ahead of the dsv4 early-return, with a new
comment stating that a draft shipping its own embeddings and head "can run on
devices the target does not use (e.g. `-devd` with a tensor-split target)".
That is the exact failure diagnosed in H11. Our DFlash2 GGUF ships neither
tensor, so it should still borrow via `ctx_other` and still fail on
`-devd CUDA0` — but this is now worth re-testing rather than assuming, and it
matters more than before, since tensor split is the new default target config.

### Rebase validated — and the gain is narrower than I first said

Re-ran both split modes on the rebased build. Tensor gained +1.0–3.6% prefill;
layer gained nothing (−0.8% to +0.3%). Decode unmoved in both.

I initially read the tensor gain as "the cuBLAS static workspace helping the
prefill path". The layer cell refutes that as stated — layer prefill is cuBLAS
GEMM too and did not move. The narrower explanation that fits both: `d9b6be07`
moved cuBLAS handles from one-per-device to one-per-device-*per-stream* and
dropped the explicit `cublasSetStream` calls, which can only pay where
concurrent streams exist. Tensor split runs both cards at 97–99% at once; layer
split serialises them. Inference from the diff plus these two cells — not
instrumented, and worth revisiting if a later mode contradicts it.

Adopting the rebase regardless: the gain lands on the config Phase 1 selected,
and the multi-GPU scheduler race fix (`84979813`) and speculative fixes come
free with it.

---

## 2026-08-25 — Phase 2 on tensor split, and the end of three hypotheses

Scope note: this session ran well past its brief. The agreed scope was to close
the mainline `-sm tensor` cell and the DFlash2 rebase, and defer the rest.
What follows went on to Phase 2, H4, H7, H8, H9 and H10. Work was stopped
part-way through a KV-quant sweep. The results below are all committed and
reproducible; the sequencing was wrong, not the data.

### Tensor + MTP is the best configuration measured

| Depth | Split | Drafter | Decode | Prefill | Acceptance |
|---|---|---|---|---|---|
| 4k | tensor | none | 20.30 | 68.0 | — |
| 4k | tensor | MTP | 21.27 | 59.1 | 40.1% |
| 16k | layer | none | 12.35 | 183.9 | — |
| 16k | layer | MTP | 19.05 | 162.9 | 77.5% |
| 16k | tensor | none | 19.75 | 199.0 | — |
| **16k** | **tensor** | **MTP** | **29.26** | **198.4** | **73.3%** |

Three findings worth keeping:

1. **MTP's speedup is a function of depth**, not a constant: 1.05× at 4k,
   1.48× at 16k, because acceptance nearly doubles over that span. H1 was
   written as though a single number existed.
2. **The drafter is free on prefill under tensor (−0.3%) and expensive under
   layer (−11.4%).** Nothing in the decode numbers predicts this.
3. **Split mode matters more than the drafter.** Tensor beats layer by ~55–60%
   with and without a drafter; the drafter is worth ~50% on top of either.

### A smoke test overstated a result by 13x, and I reported it before checking

The gating smoke run (`N_PREDICT=64`, 1 rep) showed MTP at 32.60 t/s and 90.0%
acceptance — reported as +61.9%. At `N_PREDICT=400` the same cell gives 21.27
t/s at 40.1%. The first 64 tokens of a response are its most predictable
stretch. **A short generation is not a small version of a long one for
speculative decoding** — it is a different measurement. The caveat was stated
when reporting, but the number should not have been led with.

### `internal` AllReduce has never run on this rig

`ggml_cuda_ar_pipeline_init` (`allreduce.cu:405`) requires cc ≥ 700 —
`__nanosleep` is Volta+. P100 is cc 600, so `GGML_CUDA_ALLREDUCE=internal`
always falls back to the meta-backend butterfly, which is also what `none`
selects. That is why Phase 1 measured the two identical to 0.05%: same code
path, twice.

I had previously recorded "no fallback warning was logged, so the internal
pipeline really did initialise". Wrong inference — those were `llama-bench`
runs, and `llama-bench` does not surface backend warnings. `llama-server` logs
it on every start. **Absence of a warning from a tool that suppresses warnings
is not evidence.**

### Engine capability matrix — buun is not the superset it looked like

Chasing "tensor + MTP + TurboQuant" turned up that turbo KV types exist only in
`buun` and `pflash`, and that `buun` alone has all three features. It then lost
on every axis that matters: 12–14% behind on prefill bare, 3.9% behind on
decode with MTP, and TurboQuant itself costs 14.3% of decode to save 788 MiB.
`mainline-rebased` remains the engine.

Also corrected: `ik` *does* support MTP (`--spec-type mtp`) — I had assumed
otherwise. And `buun` defaults KV to `vbr`, not `f16`, so unpinned `buun` runs
are not comparable to anything.

### Harness defects this session

- Two `FAILED(load)` rows from passing `mtp`/`dflash2` as `--spec-type` when the
  valid values are `draft-mtp`/`draft-dflash`. Removed.
- One cell lost to editing `run-spec-placement.sh` while a background job was
  executing it — bash reads scripts incrementally, so the offsets shifted
  mid-run. The file passes `bash -n` afterwards, which makes it look like a
  phantom. Now in RUNBOOK §5.
- Three `FAILED(request)` rows from `turbo3/q8_0` being killed mid-run when work
  was stopped. Removed as kill artefacts.

### Not measured, do not assume

- **tensor + MTP + `GGML_CUDA_P2P=1`.** Both levers were measured alone, in
  different regimes. The cell was queued twice and ran neither time. The
  measured best remains **29.26 t/s without P2P**.
- `turbo3`/`q8_0` — queued, never ran.
- Phase 1's single-GPU (`none`) leg on `UD-IQ3_S` — deferred by the operator.

---

## 2026-08-25 — Prefill/TTFT research pass (no runs)

Research only, at the user's direction: *"What can we do for time to first token
and prefill speeds on these GPUs?"* No benchmarks executed; the rig stayed idle.
Output is [Research/prefill-ttft-2026-08-25.md](Research/prefill-ttft-2026-08-25.md)
plus H13–H21 and Phase 7.

**Two defects found in our own measurement record**, both of which change how
existing numbers should be read:

1. **`run-bench.sh` passes prompt lengths to `-p`, not depths to `-d`.** Every
   prefill figure in all 16 raw CSVs was taken at `n_depth = 0`. For a TTFT
   question that is the right measurement, but it means **we have never prefilled
   this model past 16,384 tokens**, and "200 t/s at 64k" was an extrapolation
   that had entered conversation as a measurement.
2. **Every prefill number is at `-ub 512` / `-b 2048`, the defaults.** Never
   swept. No prefill figure in this repo is tuned.

**Three findings from the source tree**, which matter more than anything found
externally:

- The target model is a **hybrid, not a transformer** — `full_attention_interval
  = 4`, so only ~16 of 65 layers are full attention. This explains the flat
  2k→16k prefill curve that nobody had questioned, caps what sparse attention
  could ever buy, and makes 100k f16 KV only 6.25 GiB.
- `gated_delta_net.cu:63` walks tokens **one at a time**, with the author's
  `//TODO: Add chunked kernel for even faster pre-fill` at line 180. That kernel
  runs ~49 of 65 layers and is the prime suspect for the prefill bottleneck (H18).
- **MMQ is unavailable on P100** (`mmq.cu:316`, dp4a needs cc 6.1) and
  **FP16 cuBLAS compute is already on** (`ggml-cuda.cu:1626`). Both closed at the
  source level before a run was spent on either.

**Externally:** [issue #25593](https://github.com/ggml-org/llama.cpp/issues/25593)
— sm_60 is wrongly routed into the FP16 fast path, flipping ~1 in 20 greedy
tokens, and **`buun` has merged the fix while our `mainline-rebased` has not**.
Every cross-engine *quality* comparison in this repo is therefore confounded;
throughput comparisons survive, the fix being throughput-neutral. This also
bears on the "PFlash is too lossy" verdict, which was formed with the bug live.

**PFlash is closed as a product** — it requires sm_80+ and there is no v2 — and
**reopened as a technique** (H21): its sm_80 dependency is concentrated in the
kernel that accelerates the *drafter*, not the one that saves the target's work.

Also: the cards are **power-capped at 175 W of a 250 W default**, which is a
prefill-specific tax since prefill is clock-bound and decode is bandwidth-bound.
Logged as H15 and **not acted on** — power work above 175 W remains deferred
pending an explicit decision.

**Recovered prior data:** `/root/niah_test/` holds an unreferenced 8k→100k NIAH
sweep from 2026-07-16 (Qwen3.5-9B, 12/12 needle hits per engine at every tier).
It is the only long-context scaling data measured on these cards and now anchors
the pessimistic end of the 64k/100k bracket. Note its `pflash_turbo3` rows are
*not* PFlash working — `run_engine.py` passes no `--pflash-*` flags at all.

Docs updated: README (Phase 7 + the open TTFT problem), HYPOTHESES (H13–H21),
METHODOLOGY (hybrid architecture, the length-vs-depth defect, the hardware
floor, the power cap), RUNBOOK (Phase 7 ordering, nine new closed lines).

## 2026-08-25 — H14 ubatch sweep: +63% prefill; PFlash closed; 220 W approved

**Two benchmark runs, both committed** (`22ca967` ok, `431eb0a` FAILED-by-design).

Ran the first `-ub` sweep in the project's history — the batch-sweep phase did not
exist before today, it was created as H14 / Phase 7 cell 3 in yesterday's research
pass. New script `scripts/run-ubatch-sweep.sh` puts the whole sweep in **one**
`llama-bench` invocation so the 4–8 min model load is paid once; both runs together
took 705 s against a 30 min budget. Peak 68/69 °C against the 83 °C limit.

**Result: `-ub 2048` is worth +63% on prefill** (357.5 vs 218.9 t/s at p=2048).
Sweep 2 at `-b 8192` found the plateau: 2048 → 342.9, 4096 → 347.5, 8192 → **OOM**
in `ggml_cuda_pool_vmm::alloc` under `ggml_cuda_mul_mat_cublas_impl`. That abort is
a legitimate result — it locates the ceiling — and was committed as a failed run per
the standing protocol. Chose `-ub 2048`: 4096's extra 1.3% is not worth doubling
activation memory that must coexist with a 6.25 GiB KV cache at 100k.

The curve is **non-monotonic** — `-ub 256` (249.7) beats `-ub 512` (218.9) at both
prompt lengths, stddev < 0.15. The default sits in a dip. Not explained; flagged.

**Three consequences.**
1. Every prefill number in Phases 1–6 was taken at `-ub 512` and is ~35% low. Still
   valid as relative engine comparisons; not valid as this rig's capability.
2. 347 t/s = 18.8 TFLOP/s = **52% of FP16 peak**, up from 30%. The research doc
   predicted 45–55% as the realistic ceiling for *all* config tuning combined, so
   one flag has claimed most of the remaining rate headroom. Further rate tuning is
   now low-yield; the work-reducing levers (prompt-cache reuse, fewer tokens,
   smaller model) matter more.
3. H18's GDN-bound thesis is weakened, not settled. H14 was designed so a null
   result would support H18; we got the opposite. Re-run H18 at `-ub 2048`.

**PFlash closed outright** by user instruction — product, fork and technique. H21
withdrawn one day after being opened. Also corrected a claim this repo was carrying:
the earlier "PFlash is too lossy" verdict was formed on a **7900 XTX**, not on these
P100s, so the sm_60 arithmetic bug (H17) never confounded it — which was the entire
argument for reopening the technique.

**H15 power cap approved to 220 W** (not 250 W). 175 W was conservative, not
measured. One human gate before any run above 175 W: PSU wall draw must be checked
with a plug-socket meter in person, as it is not readable from this host. When
cleared: step 175 → 200 → 220, one step per run, temperature log as primary output.

### Later the same day — chunked GDN research, H16 withdrawn, objective recorded

**Research only, no runs.** User: "TurboPrefill is a no-go", power cap is testable
but needs careful manual involvement at every step, and "look further into Chunked
GDN kernel". Full write-up in `Research/chunked-gdn-2026-08-25.md`.

**The main finding reverses this morning's costing.** The previous research pass
called a chunked GDN kernel "the largest lever we control" and priced it as a
from-scratch CUDA build. It is not one. `build_delta_net_chunking()` in
`src/models/delta-net-base.cpp` **already implements the chunkwise-parallel form**
as a graph of generic ggml ops (10 × `ggml_mul_mat`, plus `solve_tri`/`tri`/
`cumsum`, `CS = 64`), and `build_delta_net()` selects between it and the fused
kernel on `cparams.fused_gdn_ch`. That flag defaults true and is auto-resolved by a
probe that asks only whether the backend *supports* the fused op — which sm_60
does, so we always get the sequential kernel. There is no CLI flag. Forcing the
chunked path is a ~2-line patch. Opened as **H22**.

**Every op the chunked graph needs runs on sm_60** — `ggml-cuda.cu:5307-5311`
returns true for `CUMSUM`/`TRI`/`DIAG`/`SOLVE_TRI` with no `cc` comparison, and
none of `solve_tri.cu`/`tri.cu`/`cumsum.cu` has a `GGML_CUDA_CC_*` gate.
`solve_tri.cu`'s `MAX_N_FAST 64` matches `CS = 64` exactly.

**Why the fused kernel is the suspect, stated properly this time.** Its grid is
`H × n_seqs × ceil(S_v/4)` and contains **no `n_tokens`** — 1536 blocks for our
model (`S_v=128`, `H_v=48` from `ssm.state_size` / `ssm.inner_size`) whether the
prefill is 512 tokens or 100,000, with tokens as a sequential `for` loop inside
each block, two serial warp reductions deep, in scalar FP32. **No GEMM, no FP16**,
so 49 of 65 layers use none of the 18.7 TFLOPS the floor argument rests on. The
recurrence is only ~0.3% of prefill FLOPs, so any material *time* share implies an
efficiency two to three orders of magnitude below the GEMMs.

**H14 raised H22's value rather than lowering it.** The fused kernel is invariant
to `-ub`, so this morning's +63% came entirely from the GEMM side; Amdahl then makes
GDN a larger share of what remains (20% → ~29%, or 35% → ~47%). Corrects this
morning's note that H14 had weakened the GDN thesis — it weakened GDN as the
explanation for the *old* 30%-of-peak figure, but strengthened it as a target now.

**The best reason to expect failure**, recorded so it is not rediscovered: the
analogous Mamba-2 chunked path is gated `cc >= GGML_CUDA_CC_TURING`
(`ssm-scan.cu:829`), added whole in PR #22675 with only "Requires NVIDIA Turing+
otherwise fallback to scan" and **no stated reason**. If that encodes a performance
finding, it predicts H22 fails on Pascal. Two further costs: the unfused path
materialises q/k at 3× width (`H_k=16 ≠ H_v=48`, `qwen35.cpp:441`), and ~50 ops per
layer instead of 1 — though intermediates scale with ubatch, not context.

**Upstream:** RFC #22967 asked for exactly this CUDA kernel and was **closed
without landing**; PR #24561 is CDNA-only; PR #20377 (Vulkan) is open; FlashQLA is
Hopper. Nobody is going to hand us a chunked CUDA GDN kernel.

**H16 (TurboPrefill) withdrawn** by user instruction. Its own reasoning already
made it unattractive: ~1.3–1.6× on two cards, bought by forcing `-sm layer`, which
costs 35% of decode.

**H15 (power cap) confirmed as a live lever** but requires the user's careful
manual involvement at each step, on top of the PSU plug-meter gate already recorded.

**The objective is now written down** in README ("The objective") and RUNBOOK, at
the user's direction — it was implicit before. The deliverable is **a single server
launch command**, not a benchmark table. Four targets simultaneously: 100k context,
highest decode, highest prefill, and output quality indistinguishable from baseline.
The quality target is the binding constraint and currently has the least data.
Also recorded: intended deployment is a long-running autonomous harness
(prime-agent, or anything with a `/goal` mode) so a large prefill amortises across
a session — which is why H19 (prompt-cache reuse) is the highest-value practical item.

**Deferred by user:** the full batch × context sweep, to be done another time. Today
covered only 2k/4k/8k.

**Paused here at user request.** Next action when work resumes: Phase 7 cell 1
(H13, the 100k curve at `-ub 2048`), then cell 2 (H18) to size H22.

## 2026-08-26 — H13: the 100k curve, and the +63% doesn't survive to depth

One run, `scripts/run-ubatch-sweep.sh`, `UB=2048`, `-p 16384,65536,100000
-n 0 -r 1`. 1743 s, peak 70 °C, `results/raw/h13-prefill-depth-q4km.csv`.
Committed and pushed (`6d10bb8`).

| depth | t/s | TTFT |
|---|---|---|
| 16k | 327.1 | 50 s |
| 64k | 250.1 | 4.4 min |
| 100k | 215.4 | **7.7 min** |

**H13 confirmed** (≥150 t/s at 100k) but the assumption it was framed against
was wrong: prefill is not flat with `-ub 2048` at depth, it falls 34% from
16k to 100k. H14's 8k gain (+63%) has mostly evaporated by 100k — 7.7 min
lands near the *best case* of the old `-ub 512` bracket (7.8–15.9 min) that
H14 was supposed to beat.

**Opened H23** from this result: a GDN kernel's cost is linear in tokens and
cannot on its own produce *falling* t/s with depth — something super-linear
is responsible, and the arithmetic (GGUF: `embedding_length=5120`,
`head_count=24`, `key_length=256` → attention width 6144) says the ~16
full-attention layers' O(n²) cost is a plausible fit: ≈2.0 PFLOP at 100k,
which alone would be ~11% of aggregate FP16 peak — lower than the "31% of
peak" figure the standing linear FLOP estimate implies, and consistent with
super-linear growth. **Not measured, arithmetic only — flagged as such in
H23.** If it holds, it reverses README's "sparse attention has a low ceiling
here" claim, which was argued from layer-count share (16/65) rather than
FLOP-count share. Also revises H18: the GDN kernel and the attention
quadratic term are probably both real, at different depths — GDN-bound
short, attention-bound long. Next test for either is the same one: `nsys`
per-kernel timing on a 100k prefill.

**User: "no more tests today, document and commit."** All of the above
written into HYPOTHESES.md (H13 result, H18 update, new H23) and README.md
(TTFT table, hardware-floor and sparse-attention bullets). Paused here.

## 2026-08-26 — H20 replaced (9B -> 27B IQ3), H22 tested: crashes on tensor split

**H20 reframed at user request.** The 9B-on-one-card fallback defeated the
point of the project — a different model isn't a fallback for "serve the
27B". Replaced with Qwen3.8-27B-IQ3 on a single card: same model, aggressive
quant, same single-GPU footprint the 9B version was chasing. Not started.

**H22 run — one test, as requested, not a sweep.** Patched
`dflash2-rebased/src/llama-context.cpp`: `getenv("LLAMA_GDN_FORCE_CHUNKED")`
forces `cparams.fused_gdn_ch = false` after the auto-resolve block (has to go
after — auto-resolve overwrites the value from its device probe regardless of
what it was set to earlier). Env-gated, off by default, same pattern as the
file's existing `LLAMA_GRAPH_REUSE_DISABLE`. Rebuilt `llama-bench` only
(incremental, ~1 min). Not committed to the fork — local working copy, not
pushed, per that repo's own AGENTS.md.

One run: `-ub 2048 -b 2048 -p 16384 -n 0 -r 1 -sm tensor`, chunked path
forced. **Crashed in graph allocation before the first token:**
`GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN)` in
`ggml_backend_meta_get_split_state`. The chunked graph's ops aren't wired
with split-axis metadata for `-sm tensor` — a different failure than the
anticipated "loses without tensor cores" outcome; it never got far enough to
measure that. Peak 50 °C, aborted in under 30 s, no thermal event.

Doesn't settle whether the chunked algorithm is actually faster on sm_60 —
only that it's unwired for our specific two-card split mode. Recorded in
H20 and H22 in HYPOTHESES.md; not chased further today.

## 2026-08-27 — H24: `-ub 2048` costs 6.7% of decode, and it is the drafter's fault

Ran the matched two-arm decode test the README has been carrying a warning about
since H14. `scripts/run-h24-ubatch-decode.sh` — two server loads, identical in
everything but `-ub`, `-sm tensor`, `ALLREDUCE=none`, MTP-Q4_0, 16k prompt,
400 tokens, r=3. Both arms went to plan; peak 68 °C.

| | `-ub 512` | `-ub 2048` |
|---|---|---|
| decode | 29.39 t/s | 27.41 t/s (−6.7%) |
| prefill | 198.5 t/s | 310.5 t/s (+56.4%) |
| acceptance | 73.3% | 66.4% |
| VRAM/card | 10,235 MiB | 10,973 MiB |

The hypothesis said <5%, so it is refuted — but the interesting part is where the
6.7% comes from. Acceptance was written into H24 as a **control**: ubatch should
not be able to move it. It moved 6.9 points. Following that thread accounts for
the entire result: 400 tokens cost `400 − accepted` target forward passes, which
is 126 steps at 512 and 135 at 2048, +7.1% of work, predicting −6.7% decode.
Measured −6.7%. There is no residual — **the decode graph itself is unaffected by
ubatch**, exactly as the null reasoning predicted; the drafter simply proposes
worse when its own prefill is chunked differently.

Checked the obvious follow-up rather than assuming it: the 400 generated tokens
are **byte-identical between the two arms** and across all three reps in each
arm. So this is an efficiency change, not a quality change — the target model
corrects the worse drafts to the same output, which is what speculative decoding
is supposed to do.

Kept `-ub 2048`. At 16k it saves 29 s of prefill per turn and costs 2.45 ms per
output token; break-even is ~11,800 tokens in one turn, which no coding turn
reaches. A drafter-free config should take `-ub 2048` for free.

Incidental but useful: the standing 29.26 t/s headline reproduced at 29.39 t/s
under `ALLREDUCE=none` with identical VRAM and identical 73.3% acceptance. That
figure was measured at `ALLREDUCE=internal` and had never been re-taken since
the switch, so it was being quoted across a config change. It survives.

Two small pieces of infrastructure: `run-spec-placement.sh` gained `BATCH`/`UB`
env knobs (empty = llama-server defaults, so every earlier arm is unchanged), and
`scripts/run-h24-ubatch-decode.sh` drives the pair.

## 2026-08-27 — H20: finishing the 9B -> IQ3_S reframe, and a claim that did not survive it

The 2026-08-26 reframe changed the H20 section body but left the 9B version
standing in the places that are actually read first: the HYPOTHESES summary
table, RUNBOOK Phase 7 cell 6, the "reuse what exists" note, and the
prefill-TTFT research doc's lever table and hypothesis table. All corrected —
the single-GPU arm of this project is `Qwen3.8-27B-UD-IQ3_S` everywhere now, and
the 9B survives only as July NIAH history in H13's decay reasoning. (The rest of
RUNBOOK already named IQ3_S as the single-card target; H20 was the outlier.)

Rewriting it properly surfaced a claim that should not have been carried across:
the old hypothesis promised **≥2× prefill**, and that was a property of the 9B,
not of running on one card. A 9B cuts the FLOP count roughly 3×. IQ3 does not —
same 27e9 parameters, same arithmetic, weights dequantised to f16 before the GEMM
either way. IQ3 buys **bytes, not FLOPs**, so on one card prefill should get
*worse* (same work, 18.7 TFLOPS instead of 37.4). The hypothesis is now about
single-card viability and about decode, where halving the weight bytes and
removing all cross-GPU traffic actually helps.

This also demotes H20 in the Phase 7 framing paragraph, which listed H19 and H20
as the only remaining levers that reduce the amount of work done. H20 no longer
reduces work. H19 is now the only one.

## 2026-08-27 — H25: the IQ3_S single-card fallback, and a KV-quant prior that was wrong

Three single-GPU arms at <=16k (`scripts/run-h25-iq3-single.sh`). Sized under the
new run-economy rules: `-sm` not swept (a no-op on one card), the two-card
control arms cited from H13/H24 instead of re-run, `-ub 512` cut as settled, reps
at 2. Roughly 18 min of one card instead of the ~45 the first design would have
burned across two.

**Single card vs the pair at 16k:** prefill 184.1 vs 327.1 t/s (**56%**), decode
drafter-free 10.4 vs 19.75 t/s (**53%**). Against the actual working config with
MTP (27.41 t/s) the fallback is at **38%**.

**MTP does not fit on one card**, and that is the finding that decides H20. The
drafter loaded at 16,029 MiB of 16,269 and then OOM'd in `ggml_cuda_pool_vmm::alloc`
on the first request — 240 MiB is not enough for the `-ub 2048` compute buffer.
This was already with q8_0 KV, so the obvious mitigation was spent. The fallback
loses the +48% decode lever, which is why the honest gap is 38%, not 53%.

**The unexpected result: KV quantisation is nearly free on this rig.** At `-ub 2048`
on one card, pp16384 is 183.80 (f16) / 184.13 (q8_0) / 184.05 (q4_0) — inside
0.2% — and tg128 is 11.38 / 11.21 / 11.18, so **1.5% of decode**. q4_0 buys
nothing over q8_0.

The repo has been carrying H4's **14.3% decode cost** as the price of KV
quantisation. That number was measured on **TurboQuant**, and has been quoted
since as though it applied to KV quantisation generally. It does not. Stock
`q8_0` costs 1.5%.

**This matters for the two-card 100k target, not just the fallback.** q8_0 should
halve the 100k KV cache from 6,250 to ~3,125 MiB for ~1.5% of decode — the
cheapest headroom available, refused until now on the strength of a number
measured on a different technique. Corrected in H4 and H25.

Incidentally the run reconfirmed H24 from the other direction: drafter-free
decode is 11.41 t/s at `-ub 512` and 11.38 at `-ub 2048`. A 4x ubatch change
moves decode 0.3% once the drafter is out of the picture, which is exactly the
null H24 predicted.

**Follow-up offered and declined:** whether MTP fits on one card at a smaller
`-ub`. Not parked — closed with the rest of the single-GPU line (see below).


---

## 2026-08-27 — Closing the single-GPU line, and what today actually bought us

User verdict, verbatim in effect: *"we've established the baseline for a single
GPU and it's not feasible for real world use."* H20 and H25 are **closed, not
parked** — parked means the question may still be worth power later; this one is
answered. Recorded in HYPOTHESES.md (both sections and both summary rows),
RUNBOOK Phase 7 cell 6 and the parked table, README's Phase 7 row, and
RESULTS.md, which gained a Phase 7 section carrying the H24 and H25 tables it
had been missing.

### The day in four results

**H24 — `-ub 2048` costs 6.7% of decode, and it is not the ubatch's fault.**
27.41 vs 29.39 t/s, against +56.4% prefill. The acceptance control moved (73.3
-> 66.4%), which normally voids a test; instead the token-accounting identity
closed it exactly — 126 vs 135 target forward passes is +7.1% work, predicting
-6.7% against -6.7% measured, no residual. There is no intrinsic ubatch penalty
on the decode path. Break-even at ~11,800 output tokens, so `-ub 2048` stays.

**H20's inherited claim did not survive its own reframe.** Finishing the 9B ->
IQ3_S rewrite turned up a "at least 2x prefill" line that was a property of the
9B having 3x fewer FLOPs, not of running on one card. IQ3 buys bytes, not FLOPs
— weights dequantise to f16 before the GEMM — so prefill scales with card count
regardless of quant. Deleted, not carried forward.

**H25 — the fallback is at 38% of what we actually serve.** 56% of the pair's
prefill, 53% of its drafter-free decode, and MTP OOMs outright at 16,029 of
16,269 MiB with `q8_0` KV already spent. Two independent failures against the
objective: it cannot use the decode lever and it cannot reach 100k. Quality was
never measured and now never will be.

**The one result that outlives the line of enquiry: KV quantisation is nearly
free.** 0.2% of prefill, 1.5% of decode, and `q4_0` buys nothing over `q8_0`.
The repo had been quoting 14.3% — TurboQuant's own number, generalised to all KV
quantisation on no evidence. This is about the **two-card** config: `q8_0`
should halve the 100k KV cache from ~6,250 to ~3,125 MiB for ~1.5% of decode.
Cheapest headroom on the table, and it was refused for weeks on a wrong prior.

### Process changes made permanent today

Rule zero — **"do we really need to test this?"** — is now a hard limit in
CLAUDE.md and RUNBOOK §2.1, alongside three design rules: prefer many short runs
to few long ones; never re-measure a settled lever (cite the banked result as
the control); and abort early, with the skip condition built into the driver so
an unattended sweep stops itself. H25 was the first run designed under them —
cut from 6 arms to 3, f16 KV dropped, reps 3 -> 2, and shipped with an
early-abort that would have skipped the q4_0 arm had q8_0 failed.

Applied retroactively: H18, H22, H23, H12 and the q8-KV-x-ubatch question are
parked with written reasons so nobody re-spends the power on them.

### What is left that can change the serve command

The 100k NIAH quality gate (the fourth objective still has zero data), H19
(prompt-cache reuse, a flag not a sweep), H17 (the sm_60 arithmetic bug that
confounds every quality comparison), H15 (power cap, gated on the user's
in-person PSU meter check), and now **q8_0 KV on the two-card 100k config** —
newly cheap, and not yet run.

---

## 2026-08-27 — H26: the config actually serves at 100k

**The rig serves the target depth.** `-c 100000 -ub 2048 -sm tensor -ctk q8_0
-ctv q8_0` with the MTP drafter: **20.02 t/s decode, 207.4 t/s prefill, 60.3%
acceptance, 12,147 MiB/card, 70 C peak, ~7.8 min TTFT.** Two reps, arm A only —
it fit, so the `q4_0` and no-drafter fallbacks never ran.

Until today every serve-side number in this repo was 16k. H13's `pp100000` was
`llama-bench`: no drafter, no server, no decode, no VRAM, no thermals. The
config had never been shown to *load* at 100k, let alone run.

### Five gaps closed on one request

1. **It fits** — 12,147 MiB/card, ~4.1 GiB spare. The pre-run estimate was
   12.0–12.5 GiB, so the memory model is sound. `q4_0` KV withdrawn as
   unnecessary.
2. **Decode at 100k: 20.02 t/s** — 73% of the 16k figure.
3. **Acceptance at 100k: 60.3%** — and it is **not monotonic in depth**: 40.1%
   at 4k, 66.4% at 16k, 60.3% at 100k. It peaks in the middle. No drafter
   conclusion transfers across depths in either direction.
4. **`q8_0` KV works on two cards**, as H25 predicted from one card. The 100k
   cache cost ~1,174 MiB/card against ~2,350 at f16.
5. **Sustained thermals are a non-issue** — 70 C across two consecutive
   ~8-minute prefills, identical to the 3-minute 16k runs, 13 C under the abort
   limit. The 175 W cap is doing the work. This was the run's real risk and it
   did not materialise.

**Cross-harness agreement, checked for the first time.** `llama-server` gave
207.4 t/s prefill at 100k against `llama-bench`'s 215.4 — 96% agreement on
different days with different tooling. The gap is about what the chat template
and HTTP path should cost.

### The hypothesis missed its own threshold

H26 claimed decode would stay within 25% of the 16k figure. It is **26.9%**
down. Recorded as a miss rather than rounded away — but the question the run
existed to answer is settled, and the answer is yes.

### A harness bug that looked like a model failure

The first attempt loaded fine and then returned HTTP 500 with
`parse error ... attempting to parse an empty input` on both reps. Nothing to do
with VRAM or the model: the ~340 KB prompt was passed through **argv** to
`python3 -c` and then to `curl -d`, which overflows **E2BIG**, so curl posted an
empty body. Worse, `run-spec-placement.sh` returned 0 after every rep failed, so
the arm ladder cheerfully printed "ARM A SUCCEEDED" over two failures.

Both fixed: the request body is built once to `logs/<label>.payload.json` and
posted with `-d @file`, and the script exits non-zero when no rep produces a
measurement. The two `FAILED(request)` rows are kept in the CSV — they are a
real recorded failure and the `outcome` column explains them. Cost: ~4 minutes
of GPU, caught on the first arm.

**Lesson worth keeping:** a 500 from the server is not evidence about the model
until the request that produced it has been shown to be well-formed.

### A predicted limitation that did not appear

I expected rep 2's prefill to be contaminated by slot-level LCP prefix reuse —
the trap that invalidated the whole H11 prefill column. It was not: 207.13 vs
rep 1's 207.71, so prefill was genuinely re-done and both reps are valid. The
H11 trap is therefore not universal. **Cause unconfirmed** — `-np 1` with
`kv_unified = false` is the obvious suspect, but this run did not isolate it, so
`cache_prompt: false` still should not be trusted on its own.

### What this leaves

Three of the four objectives now have numbers **at the target depth, from a
single run**. The fourth — quality — still has none, at any depth, and is now
the only thing between this repo and its deliverable. The next question is not
how fast this rig is; it is whether what it produces at 100k is worth reading.

---

## 2026-08-27 — H17: the sm_60 FP16 fix is real, and it is not free

**Refuted as written.** Patching sm_60 out of the FP16 fast path costs **12-13%
of prefill** and gains **+0.36%** of decode, against a claim of "throughput-free"
and "+1.4% decode". The adopt-for-free path H17 assumed does not exist.

| | control | patched | delta |
|---|---|---|---|
| pp2048 | 222.72 | 193.01 | **-13.3%** |
| pp16384 | 208.44 | 182.76 | **-12.3%** |
| tg128 | 20.357 | 20.430 | +0.36% |

Paired A/B, same tree, same flags, one session, std devs 0.01-0.14 t/s. Peaks
69 C and 70 C. The build itself cost no GPU time.

### Why "free by hypothesis" was wrong

sm_61 has the carve-out because GP104 runs FP16 at **1/64 rate**, so excluding it
costs nothing. **GP100 runs FP16 at 2:1** — 18.7 against 9.3 TFLOPS. The patch
therefore moves real work off a genuinely fast path that sm_61 never had. One of
the five consumers is `ggml_cuda_mul_mat_cublas`, which flips the cuBLAS compute
type from F16 to F32 **for quantized `src0`** — our Q4_K_M weights, on the prefill
critical path. That was visible in the source before the run and should have been
caught when H17 was written.

### The pre-registered prediction paid off, and it corrects a banked result

Written down before the run: if the fix explains buun's prefill deficit, the
patched build converges on buun's 193.50.

**Measured 193.008 at pp2048 and 182.76 at pp16384, against buun's 193.50 and
182.75 — within 0.3% at both depths.**

So **buun's 12-14% prefill "deficit" was never an engine difference.** buun
carries the fix; rebased does not; that is the whole gap. Since Phase 3 this repo
has been reading an arithmetic-mode difference as an engine property. RESULTS.md
is corrected. Like for like the two engines have the same prefill, and buun's
**decode** lead shrinks to ~1.2% and is real.

The decode half of the prediction **failed** — predicted +1.6%, measured +0.36%.
Recorded as a miss.

### H18 got its number as a side effect

Halving the theoretical GEMM rate cost only 12-13% of prefill. Modelling prefill
as a GEMM-rate-bound fraction `f` plus everything else, `1/(1+f)` = the measured
ratio gives **f = 15.4%** at 2048 and **14.1%** at 16384.

**~15% of prefill time is GEMM-rate-bound; ~85% is not** — direct support for
H18's claim that the GDN path, which has no GEMMs, is where prefill time goes.
H18 was parked in the morning for want of a cheap probe. It got one for free.

The estimate assumes cuBLAS FP32 is exactly half-rate and that nothing else
changed; the FP32 path also converts weights to F32 rather than F16, so some of
the 13% is bandwidth, not FLOPs. It bounds the GEMM share, it does not measure
the GDN share.

### Not adopted, not rejected

At the H26 serve config, -12.3% prefill takes 100k TTFT from **~7.8 to ~9.2 min**
per uncached turn. Against that, upstream reports KL divergence
0.004962 -> 0.000001 and top-token agreement 95.00% -> 99.89%: **about 1 greedy
token in 20 flips**.

**We did not verify that on this rig.** The output-comparison test was planned
and not run - the user called a halt to further testing, correctly, since the
throughput number already forced the decision up a level. So the cost side is
measured and the benefit side is still upstream's number.

This cannot be settled on speed alone. It is a speed-versus-quality trade and it
belongs with Phase 8, where quality finally gets measured - **which also means
H17 must be settled before the quality sweep runs, not after, because it changes
model numerics.** Parked for a decision.

**One option recorded so nobody re-derives it:** the three patch sites are
separable. `fast_fp16_hardware_available` alone sets the cuBLAS compute type for
prefill GEMMs; the other two govern the decode mat-vec, flash-attention tile and
quantized dot products. Patching only the latter two would keep FP16 prefill and
move just the decode-side arithmetic to FP32 - recovering most of the 13% while
still fixing the paths nearest the sampled logits. Whether that keeps the
accuracy benefit is unknown; it would need one free build and one short run.

### Artefacts

`patches/h17-sm60-fp16-fastpath.patch` (three sites, copied verbatim from buun),
`/root/dflash2-rebased/build-h17` (patched binaries; the H26-validated
`build-cuda-p100` was left untouched), `scripts/run-h17-ab.sh`, and
`rebased-h17` as an engine in `run-bench.sh`. The source tree itself is reverted
to unpatched, so a rebuild of `build-h17` must re-apply the patch first.
