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
