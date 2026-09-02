# Runbook — operating procedure

The procedure for executing benchmark runs on this rig. Follow it in order. It
assumes no prior context beyond what's in this repo. §7 is the work queue.

---

## 0. Non-negotiables

1. **Never let a card exceed 83 °C.** Abort the run if it does. Hard shutdown
   is 87 °C on passively cooled cards behind a custom fan shroud.
2. **Every run costs the user real electricity.** Prefer many short runs to
   few long ones, never re-measure a settled lever, abort a sweep as soon as
   its conclusion is clear. §2.1 is the full rule and it is a hard requirement.
3. **Commit and push after every run, pass or fail** (§6). A failed run is
   data; losing it wastes 5–20 minutes of load time to rediscover.
4. **Don't change two variables at once.** One lever per run.
5. **Don't silently change fixed parameters** in [METHODOLOGY.md](METHODOLOGY.md).
   If a run needs different ones, record why in [RUNLOG.md](RUNLOG.md).

---

## 1. Preflight

```bash
nvidia-smi -L                       # expect exactly 2× Tesla P100-PCIE-16GB
nvidia-smi --query-gpu=index,temperature.gpu,power.limit,persistence_mode \
           --format=csv             # expect ~45–50 °C idle, 175 W cap, persistence Enabled
df -h /root                         # models are 11–22 GB each
```

Power cap and persistence do not survive a host reboot (the cards revert to
250 W). Restore with:

```bash
nvidia-smi -pm 1
nvidia-smi -pl 175 -i 0 && nvidia-smi -pl 175 -i 1
```

**Power cap gate (H15).** 175 W is the baseline for every result in the repo.
Raising it is approved to **220 W, not 250 W**, but only after the user has
measured PSU wall draw with a plug-socket meter in person — it is not readable
from this host. When cleared, step 175 → 200 → 220, one step per run, and
record the cap as a column. Never change it unprompted.

**Check for competing disk I/O.** Model loads are disk-bound; a download or
copy makes them 2–3× slower. Wait, or note it in RUNLOG.

Never edit `scripts/*.sh` while `pgrep -f 'llama-server|llama-bench'` shows
work in flight — bash reads scripts incrementally and the running job corrupts.

---

## 2. Running one benchmark

**`llama-bench`** (Table A — engine, split, quant, ubatch, depth):

```bash
cd /root/p100-benchmarks
./scripts/run-bench.sh <engine> <model-path> <split-mode> <label> [extra llama-bench args...]
# e.g.
./scripts/run-bench.sh mainline /root/Qwen3.8-27B-UD-Q4_K_M.gguf tensor h99-something -ub 2048
```

**`llama-server`** (Table B — drafters, acceptance, VRAM, real requests):

```bash
SPLIT=tensor GGML_CUDA_ALLREDUCE=none CTX=16384 UB=2048 BATCH=2048 REPS=3 \
  ./scripts/run-spec-placement.sh <label> <draft-mtp|draft-dflash|none> <drafter.gguf|none> <default|CUDA0|CUDA1>
```

Env knobs (`MODEL`, `BIN`, `CTK`/`CTV`, `N_PREDICT`, `PROMPT_FILE`, `NP`, …)
are documented in the script header. `scripts/run-h26-100k-serve.sh` is the
worked example of a ladder that stops itself.

- `<engine>`: `mainline` (rebased) | `buun` | `ik`. Split modes differ per
  engine — METHODOLOGY §2.
- `<label>`: filenames and commit message. `h<N>-<what>-<quant>` or
  `phase<N>-<engine>-<split>-<quant>`.
- Both scripts do the temp preflight, telemetry, logging, CSV append, and the
  commit/push. Launch them in the background and wait for completion; don't
  poll with sleep loops.

### 2.1 Run economy — every run costs electricity

A two-card run draws ~350 W for its whole duration; a 100k prefill sweep is
30+ minutes of it.

**Rule zero: "do we really need to test this?"** Ask it of every arm, sweep
and open hypothesis. A test earns its power only if it can **change what goes
in the serve command**. Stop at the first "no":

1. **Do we already know the answer?** Search RESULTS.md and `results/` first
   and cite the banked number as the control.
2. **Would either outcome change the serve command?** If not, it is a
   diagnostic. Write the reasoning down instead of running it.
3. **Is there a cheaper way to the same confidence?** Arithmetic, a source
   read, a shorter proxy run.
4. **Is this the smallest run that answers it?** Cut depths, reps and sweep
   points that don't change the verdict.

A hypothesis that fails rule zero is **parked with the reason** in §7, never
silently dropped. Then, before launching:

- **Many short runs over few long ones.**
- **Never re-run a settled lever.** If a control exists in `results/`, cite
  it; don't re-take it for symmetry.
- **Abort early**, and **build the skip condition into the driver script** so
  an unattended run stops itself (`run-h25-iq3-single.sh` skips the `q4_0` arm
  when `q8_0` fails; `run-h26-100k-serve.sh` is a ladder).

Not licensed: dropping a control the repo lacks, or cutting reps below 2. When
you trim an arm, say in RUNLOG what assumption the saving rests on.

### 2.2 Expected timing — read before assuming a hang

| Phase | Duration | What it looks like |
|---|---|---|
| Model load | **4–8 min** (longer under disk contention) | Process in `D` state, GPUs idle, no output past the CUDA banner |
| Prefill sweep to 16k | 3–6 min | GPUs at 97–99% together (tensor) or alternating (layer) |
| 100k prefill | ~8 min per request, sustained | Same, for longer. Watch the temps log |
| tg128 | ~30 s | |

Confirm progress with `cat /proc/<pid>/io` (read_bytes climbing) rather than
killing a load.

---

## 3. Thermal safety

Both scripts start `scripts/gpu-monitor.sh`, which samples every 5 s into
`logs/<label>.temps.log` (`HH:MM:SS idx, tempC, watts, util%`).
`tail -f` it during any run longer than a few minutes.

| Temp | Action |
|---|---|
| ≤ 80 °C | Normal. Observed peak across the whole project is 70 °C |
| 80–82 °C | Finish the current run, don't start another until cooled |
| ≥ 83 °C | **Abort immediately** (`pkill -f 'llama-bench|llama-server'`), record in RUNLOG |
| 87 °C | Hardware shutdown — must never be reached |

Brief readings above 175 W are sampling overshoot; the cap is a rolling
average.

---

## 4. Recording results

1. **Raw data is written by the script**: `results/raw/<label>.csv` for
   `llama-bench`, a row per rep in `results/h11-placement.csv` for
   `llama-server`. Don't hand-edit either. (`results/all-results.csv` is a
   stale aggregate with mixed engine schemas — never parse it by column.)
2. **[RESULTS.md](RESULTS.md)** — Table A: run `python3 scripts/build-matrix.py`
   and paste over the table. Table B: add a row by hand (mean of reps, peak
   temp, anything invalid flagged in the notes). Table C: update the lever's
   verdict if it changed.
3. **[HYPOTHESES.md](HYPOTHESES.md)** — update the status table row and the
   hypothesis's section with the result and run label. This is the main
   reason the run exists; it is the easiest step to forget.
4. **[RUNLOG.md](RUNLOG.md)** — an entry only if something notable happened:
   a crash, a deviation, a surprise, a corrected claim.
5. **§7 below** — move the item between open / parked / closed.

Phase 5 runs additionally claim an index in [WEB_BENCH.md](WEB_BENCH.md) §5
and are hand-scored per its §4.

---

## 5. When a run fails

Do not just retry. Failures are results.

1. `logs/<label>.log` (or `.server.log`) holds full stdout+stderr and any
   backtrace. Capture the error.
2. Record it in RUNLOG with the exact string and the command.
3. Check whether it bears on a hypothesis.
4. **Commit and push the failure** before attempting a fix.
5. Only then diagnose. If the fix is a rebuild, record the commit and flags in
   METHODOLOGY §2.

### Known failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `ncclAllReduce failed with status 1` at `reduce.cu:169` (`ik`) | Built with NCCL (`GGML_NCCL` defaults ON); no NVLink | Use `/root/ik_llama.cpp/build-nonccl/` |
| `ggml-cuda.cu:106: CUDA error` on mainline `-sm tensor`, backtrace frame `ggml_backend_cuda_comm_allreduce_nccl` | NCCL AllReduce is the Linux default and is broken rig-wide | `GGML_CUDA_ALLREDUCE=none`. No rebuild |
| `internal AllReduce init failed (n_devices != 2?)` | Misleading: the real gate is sm_70 (`__nanosleep`) | Harmless; `internal` falls back to the `none` path. Use `none` |
| `error: invalid parameter for argument: -sm` | Split mode not supported by that engine | METHODOLOGY §2 support table |
| `LLAMA_SPLIT_MODE_TENSOR not implemented for architecture` | The gate is a denylist and `qwen35` is permitted — if you see this, something else changed | Check `llm_arch_supports_sm_tensor()` in `src/llama-arch.cpp` |
| `ggml-backend-meta.cpp:543: GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0)` with DFlash2 | DFlash2 borrows the target's `output.weight` via `ctx_other`; tensor split shards it on axis 0 | Not fixable by config. DFlash2 is layer-only. H8 |
| `dflash requires ctx_other to be set` / `pre-allocated tensor (output.weight) in a buffer (CUDA1)` with `-devd CUDA0` | Same borrowed tensor lives on CUDA1 under layer split | Use `default` or `CUDA1` placement. Upstream bug in PR #27342. H11 |
| DFlash2 run shows poor acceptance, no error | A v1 engine (`buun`/`ik`) silently ran the v1 path | Use `mainline`; confirm the selector in the server log. H8 |
| Abort at `fattn.cu:348` with turbo KV types | `(turbo*, F16)` dispatch unsupported | Symmetric `turbo3/turbo3`. Moot: TurboQuant is closed. H7 |
| `-ub 8192` aborts in the cuBLAS staging allocator | Too large | `-ub 2048` (4096 needs `-b 8192` for +1%). H14 |
| `failed to load model` on `-sm none` | Target too big for one card | Only `UD-IQ3_S` fits one P100. Single-GPU is closed anyway |
| `CUDA error: out of memory` in `ggml_cuda_pool_vmm::alloc` after a clean load | VRAM exhausted on the first request (seen at 16,029/16,269 MiB) | Smaller quant, `q8_0` KV, or drop the drafter |
| `FAILED(request)` rows with empty responses at 100k | Prompt passed through argv hit E2BIG; curl posted an empty body | Fixed 2026-08-27 in `run-spec-placement.sh` (prompt via file) |
| Prefill t/s implausibly high on reps 2+ (`f_sim_best = 1.000` in log) | Slot-level prefix reuse; `cache_prompt: false` doesn't defeat it | Prefill from rep 1 only, or a fresh server per rep. Decode/acceptance unaffected |
| Script dies with `break: only meaningful in a loop` or `<word>: command not found` mid-run | A `.sh` was edited while a background job was executing it | Re-run the affected cell; the file is fine (§1) |
| Chunked GDN graph aborts under `-sm tensor` | Its ops carry no split-axis metadata | Kernel work, not a flag. H22 parked |
| No `pp0` row | `-p 0` is a no-op | Expected |
| Load appears hung, GPUs idle | Normal load, or disk contention | §2.2 |
| `pi` hangs at startup (Phase 5) | Blocks on network in this environment | `PI_OFFLINE=1`, set by `run-web-bench.sh` |
| Phase 5 stage exits 124 | `STAGE_TIMEOUT` (3600 s) hit | A legitimate result; record it |
| Phase 5 `port N already in use` | Index reused, or a previous site is still hosted (by design) | Pick an unused index; WEB_BENCH §5 |
| Phase 5 server OOMs on load | `CTX=32768` beside a large model | `CTX=16384`, note it in RUNLOG |

---

## 6. Git protocol

**Mandatory, failed runs included.** The scripts commit and push at the end of
each run. For hand runs and document edits:

```bash
cd /root/p100-benchmarks && git add -A && git commit -m "bench: <label> — <outcome>" && git push
```

| Prefix | Use for |
|---|---|
| `bench:` | A run, pass or fail |
| `docs:` | Documentation with no new data |
| `build:` | Engine rebuilds, flag changes |
| `fix:` | Corrections to previously recorded data or claims |

Remote: `https://github.com/Tom1tk/p100-inference-benchmarks.git` (private).
If a push fails, keep the commit and retry; note persistent failure in RUNLOG.

---

## 7. Work queue

The deliverable is a serve command ([README.md](README.md)). Phases 0, 1, 2
and 6 are done, Phase 3 folded into Phase 8, Phase 7 has two cells left, Phase
5 has not started, Phase 8 is planned. Everything below is ranked by whether
it can change the serve command.

### Open

| # | Item | Why | Cost / gate |
|---|---|---|---|
| 1 | **H19 — prompt-cache reuse** (`--cache-ram 20000 --cache-idle-slots`, multi-turn) | The actual use case re-sends a growing prefix every turn. Also cuts the Phase 8 sweep from ~14 h to ~2 h | One multi-turn run; a flag |
| 2 | **H17 — decide the sm_60 FP16 fix** | Measured: −12–13% prefill, +0.36% decode. It is a correctness fix that changes numerics, so it must be applied or rejected **before** any quality run | Decision; needs Phase 8's first arm to see whether the output difference matters |
| 3 | **Phase 8 — quality gate** ([QUALITY-PLAN.md](QUALITY-PLAN.md)) | The only objective with no data. Sweep D (drafter hash check, ~45 min) first — it also settles the drafter re-rank at the current config, since every MTP-vs-DFlash2 comparison was at `-ub 512`/4k/layer | ~3 h GPU with H19, ~14 h without. After 1 and 2 |
| 4 | **H15 — power cap 175 → 200 → 220 W** | Prefill is compute-bound; the one lever that could move it without a kernel | **Blocked on the user's PSU meter check** (§1) |
| 5 | Phase 5 — agentic web build ([WEB_BENCH.md](WEB_BENCH.md)) | The deployment target itself | Long runs; run after Phase 8 fixes the config |
| 6 | `GGML_CUDA_P2P=1` with MTP | +2.9% decode drafter-free (H10), never measured in combination. Cheap | One Table B cell against the banked `h24-ub2048-mtp-16k` control |
| 7 | Concurrency (`-np` > 1) | Only if the harness issues parallel requests | Ask first |

### Parked by rule zero

Not closed — the question may still be true — but it cannot change the serve
command today. Re-open only on the stated unblock.

| Item | Why parked | What would unpark it |
|---|---|---|
| H18 — GDN-bound prefill | Diagnostic; sizes a prize only H22 can collect. Free evidence from H17: ~85% of prefill is not GEMM-bound | H22 becoming runnable |
| H22 — chunked GDN path | Crashes under `-sm tensor` (no split-axis metadata). A kernel project | Someone annotating the split axes |
| H23 — quadratic attention at 100k | Explains the depth decay, but every sparse-attention kernel is sm_80+ | An sm_60 sparse-attention path |
| H12 — `ik` graph-split knobs | `ik` has no tensor split and lost Phase 1 | `ik` gaining tensor split |
| H2, H3 — XL split-stall, stock-quant degradation | Fold into Phase 8 Sweep Q arms 3 and 5, only if Q4_K_M shows a gap | A gap in Sweep Q |
| q8 KV × ubatch interaction | Noted during H25; not worth its own run | An unexplained q8 result |

### Closed — do not spend runs here

| Line of enquiry | Why |
|---|---|
| Single-GPU serving (H20/H25) | 56% of the pair's prefill, 38% of its real decode, MTP OOMs, cannot reach 100k. **User call, 2026-08-27** |
| PFlash — product, fork, technique (H21) | sm_80-only, no v2, hand-port not worth it. Fork kept on disk only for `llama-niah` |
| TurboPrefill (H16) | Forces `-sm layer`, −35% decode. User call |
| TurboQuant KV (H4/H7) | −14.3% decode for 788 MiB |
| `buun` as an engine | Its prefill deficit was the H17 fix; its only exclusive feature is TurboQuant |
| DFlash2 as the drafter (H8) | Layer-only; tensor + MTP beats layer + DFlash2 by +71% decode. Still an arm in Phase 8 Sweep D |
| Q8 drafters (H9) | Accept less, same speed, +0.87 GiB |
| AllReduce tuning (H10) | Only the butterfly path runs on Pascal |
| Drafter placement `-devd` (H11) | Moves VRAM, not throughput |
| `-ub` above 2048 (H14) | +1% for double the activation VRAM; 8192 aborts |
| KV quant as a way to *reach* 100k | Unnecessary — f16 fits. `q8_0` is used for headroom, not fit |
| vLLM / vllm-pascal / SGLang / TensorRT-LLM / ExLlamaV3 / ktransformers / FlashQLA | sm_70+ minimum, or no `qwen35` support |
| MLC-LLM / TVM, tinygrad | No hybrid-SSM support; would mean rewriting the stack |
| `GGML_CUDA_FORCE_MMQ`, `GGML_CUDA_F16` | No-ops on cc 6.0 (no dp4a; F16 compute already selected) |

### Measurement rules that constrain later phases

- Quote every prefill figure with its length **and ubatch**; every acceptance
  figure with its depth. Neither transfers.
- Speculative runs: ≥400 output tokens, ≥2 reps. Never compare acceptance
  across engines without pinning `n_max`/`p_min`.
- Phase 8 freezes the serve command before its first arm and moves one lever
  per run: the model file in Sweep Q, the drafter in Sweep D.
