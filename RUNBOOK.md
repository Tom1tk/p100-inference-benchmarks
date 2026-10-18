# Runbook — operating procedure

The procedure for executing benchmark runs on this rig. Follow it in order. It
assumes no prior context beyond what's in this repo. §7 is the work queue.

---

## 0. Non-negotiables

1. **Abort any run at 80 °C.** At 83 °C a card drops off the PCIe bus and
   needs a host reboot to re-mount, and the fan controller, no longer able to
   read it, runs the fans at 100%. (Passively cooled cards behind a custom fan
   shroud.)
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
           --format=csv             # expect ~45–50 °C idle, 200 W cap (since 2026-10-15), persistence Enabled
df -h /root                         # models are 11–22 GB each
```

Power cap and persistence do not survive a host reboot (the cards revert to
250 W). Restore with:

```bash
nvidia-smi -pm 1
nvidia-smi -pl 200 -i 0 && nvidia-smi -pl 200 -i 1   # 175 for any re-run of a pre-2026-10-15 result
```

**Power cap gate (H15).** 175 W is the baseline for every result in the repo.
Raising it is approved to **220 W, not 250 W**, but only after the user has
measured PSU wall draw with a plug-socket meter in person — it is not readable
from this host. When cleared, step 175 → 200 → 220, one step per run, and
record the cap as a column. Never change it unprompted.

**Done 2026-10-15 (H15, user-set caps, 2-card serve command, REPS=2):** 16k prefill +4.5% at 200 W, +6.4% at 220 W, +6.5% at 250 W (draw never above 216 W: saturated); 98k +4.4% at 200 W; decode flat at every cap. **The user set 200 W as the running cap.** Results before 2026-10-15 are at 175 W: compare across the change only against a control at the same cap.

**Re-test at the higher cap (user call, 2026-10-14): don't discard marginal compute-side wins.** At 175 W
the 1-card arm sits at the cap ~50% of its busy time and the 2-card arm 15–30% (gpu-monitor power column,
util ≥ 90% samples). A lever that removes stalls makes the card draw more and the cap claws the gain back,
so prefill levers can look marginal here and grow at 220 W. Decode is memory/latency-bound and unaffected.
Re-test each of these with one short 16k arm per lever at the running cap (200 W; control `p15-200w-16k`). **2-card rows parked 2026-10-15 by H52:** the clock-event counters show 0 s power-capped at 200 W on every 2-card arm (16k and 98k), so the premise does not hold there. Only the 1-card H39 re-test ran (H39r). Expect little: at 200 W the 2-card arm only gained 4.5%, so the cap was not holding much back:
| Lever | At 175 W | Why it may grow |
|---|---|---|
| H39 `GGML_PASCAL_DEQUANT_STAGED=1` (bit-identical, code kept, off) | 1-card +4.5%, 2-card +1.0% (rep 2; rep 1 −3.6%) | More dequant throughput = more power in a capped phase |
| H14 `-ub 4096 -b 8192` | +1.3% prefill | Larger GEMMs, compute-bound |
| H35 ALGO3, 1-card | +1.0% (already on) | Note was "card likely at the 175 W cap" |
| H40 `GGML_PASCAL_PAR_DISPATCH=1` (bit-identical, code kept, off) | 2-card decode +0.3% | Faster kernels make host dispatch a larger share of each token |
| P5 FA tile config sweep (micro-bench only) | stock fastest, next +1.2% | Ordering of compute-bound configs can shift with clocks |

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
SPLIT=tensor GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 CTX=16384 UB=2048 BATCH=2048 REPS=3 \
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
| ≤ 75 °C | Normal. Observed peak across the whole project is 70 °C |
| 76–79 °C | Finish the current run, don't start another until cooled |
| ≥ 80 °C | **Abort immediately** (`pkill -f 'llama-bench|llama-server'`), record in RUNLOG. The four driver scripts do this themselves: `gpu-monitor.sh` with `ABORT_TEMP` set kills the driver's `llama-*` children within one 5 s sample |
| 83 °C | Card drops off the bus: host reboot needed, fans pinned at 100%. Must never be reached |

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

The deliverable is a serve command ([README.md](README.md)). Phases 0, 1, 2, 6
and 8 are done: the command passed the Phase 8 gate (H32) on 2026-10-14 and now
only changes by fork levers, each one gated (KLD vs the FP32 Q5 reference, H35
style) and measured against the banked control. Fork design notes live in the
fork repo, `docs/pascal/fork-plan-2026-10-14.md`. Ranked by whether an item can
change the serve command. Refreshed 2026-10-17. **Reopened 2026-10-17 (user call)** for Round 4 (H58–H64, from the Kmic-68 P100 fork): item 7. Item 6 stays open; item 3 closed (single card dropped).

### Open

| # | Item | Why | Cost / gate |
|---|---|---|---|
| 1 | "Re-test at the higher cap" list (§1), at 200 W | **Closed 2026-10-15:** 2-card rows parked by H52 (0 s capped at 200 W); 1-card H39 re-tested as H39r | — |
| 3 | 1-card decode: IQ3_S / IQ3_XXS matvec (~228 GB/s, ~47% of 1-card matvec time) | Biggest remaining 1-card decode share | **Closed 2026-10-17:** single card dropped (user call) |
| 4 | Phase 5: agentic web build ([WEB_BENCH.md](WEB_BENCH.md)) on **opencode v2** (replaces pi, user call 2026-10-15) | The deployment target itself; carries H51 | **Done 2026-10-17:** 2136 s, 3/3 stages (RESULTS Phase 5) |
| 2 | **Round 3 (H48–H53, HYPOTHESES):** f16 KV at 98k, draft length, 98k decode profile, power-cap counter, UD-IQ4_XS | User call 2026-10-15, each REPS ≥ 3 | **Done:** H48–H53 all closed (HYPOTHESES); H51 cache part passed in Phase 5 |
| 7 | **Round 4 (H58–H64, HYPOTHESES):** verify kernels, sampled drafts, draft-step graphs, prefill overlap, homogeneous quant, Q6_K, race audit | Kmic-68 fork shows MTP decode headroom at depth | Order in HYPOTHESES "Round 4"; each REPS ≥ 3 |
| 6 | Concurrency (`-np` > 1) | Only if the harness issues parallel requests | Ask first |

Done on 2026-10-14 and folded into the command or the 1-card arm: H19 prompt cache (defaults), H17 (closed as
negligible by H30), Phase 8 (H32), P2P with MTP (H27), the fork levers H28/H31/H34/H35/H36/H41b, and for
1-card only H42 (`GGML_PASCAL_MMVH_LOWBIT=1`, +13.0% decode). 2026-10-15: H43 (branch-free K-quant scales,
+2.1% decode). H44 (`GGML_PASCAL_MMVH_LOWBIT=1` on 2 cards, Q3_K + IQ4_NL on mmvh: +6.1% decode, output identical). H45 (drafter LM head trimmed to 96k vocab rows, `mtp-…-head96k.gguf`: +3.1% decode, output identical). H46 (`GGML_PASCAL_FA_GEMM=1`, prefill flash attention through cuBLAS GEMMs: +19.2% prefill at 98k, H32 haystack 19/19).

### Parked by rule zero

Not closed — the question may still be true — but it cannot change the serve
command today. Re-open only on the stated unblock.

| Item | Why parked | What would unpark it |
|---|---|---|
| H18 — GDN-bound prefill | **Refuted 2026-10-14** (Phase 0 op timings: GEMMs ~87% of prefill, GDN scan ~6%). Row kept for history | — |
| H22 — chunked GDN path | Crashes under `-sm tensor` (no split-axis metadata). A kernel project | Someone annotating the split axes |
| H23 — quadratic attention at 100k | Explains the depth decay, but every sparse-attention kernel is sm_80+ | An sm_60 sparse-attention path |
| H12 — `ik` graph-split knobs | `ik` has no tensor split and lost Phase 1 | `ik` gaining tensor split |
| H2, H3 — XL split-stall, stock-quant degradation | Fold into Phase 8 Sweep Q arms 3 and 5, only if Q4_K_M shows a gap | A gap in Sweep Q |
| q8 KV × ubatch interaction | Noted during H25; not worth its own run | An unexplained q8 result |
| IQ1_S on mmvh | 0.3% of 1-card bytes | Nothing bigger left on the matvec list |
| P2 chunked-GEMM comms overlap (fork plan) | Ceiling ≤ +4–5% at 16k and lower at 98k, where attention grows the compute per ubatch and comms stays fixed. Needs a side copy stream plus GEMM splitting. H36 (duplex) took the cheap half | A fresh pp2048 trace showing 2-card idle > 5% |
| H46 follow-up: GEMM-FA softmax/combine/convert | ~7 of 48 ms per FA call, ≤ ~2–4% of 98k prefill | Prefill becoming the binding target |
| H60 — CUDA graphs on MTP draft steps | Knob ablation on the reference build: ~0 on the 16k serve path (2026-10-18). H27 already found graphs neutral on plain decode | A trace showing draft-step launch gaps > 5% |
| H67 — add + rms_norm + mul fusion | Same ablation: ~0 on the 16k serve path (2026-10-18) | Launch overhead becoming the binding cost after H58 |
| P3 leftovers: dequant pass + F16↔F32 converts | H39 staged dequant: 2-card +1.0% (noise), already on the 220 W list. HGEMM is at ~90% of FP16 peak with ALGO3 (H33) | H15 (220 W) |

### Closed — do not spend runs here

| Line of enquiry | Why |
|---|---|
| Single-GPU serving on stock builds (H20/H25) | 56% of the pair's prefill, 38% of its real decode, MTP OOMs, cannot reach 100k. **User call, 2026-08-27.** Reopened 2026-10-14 for fork builds; **dropped for good 2026-10-17 (user call)**, see CLAUDE.md Scope |
| PFlash — product, fork, technique (H21) | sm_80-only, no v2, hand-port not worth it. Fork kept on disk only for `llama-niah` |
| TurboPrefill (H16) | Forces `-sm layer`, −35% decode. User call |
| TurboQuant KV (H4/H7) | −14.3% decode for 788 MiB. **Dropped from the queue, user call 2026-10-15** |
| `buun` as an engine | Its prefill deficit was the H17 fix; its only exclusive feature is TurboQuant |
| Straight Q4_0 main model | Not in mmvh, and sm_60 has no dp4a for the stock path; quality below Q4_K_M, which already failed the KLD gate (H62). Byte ceiling ~5–6%. **User call, 2026-10-17**: keep UD, speed up its types instead (H65) |
| DFlash2 as the drafter (H8) | Layer-only; tensor + MTP beats layer + DFlash2 by +71% decode. Still an arm in Phase 8 Sweep D |
| Q8 drafters (H9) | Accept less, same speed, +0.87 GiB |
| AllReduce tuning (H10) | Only the butterfly path runs on Pascal |
| Drafter placement `-devd` (H11) | Moves VRAM, not throughput |
| `-ub` above 2048 (H14) | +1% for double the activation VRAM; 8192 aborts |
| KV quant as a way to *reach* 100k | Unnecessary — f16 fits. `q8_0` is used for headroom, not fit |
| vLLM / vllm-pascal / SGLang / TensorRT-LLM / ExLlamaV3 / ktransformers / FlashQLA | sm_70+ minimum, or no `qwen35` support |
| MLC-LLM / TVM, tinygrad | No hybrid-SSM support; would mean rewriting the stack |
| Main-type matvec (Q4_K/Q5_K/IQ4_XS) tuning after H43 | Occupancy, unroll, conversions, wave tail and IQ4 lookup all probed flat (fork plan, 2026-10-15). ≤ ~3% decode left at this kernel structure |
| fp16-accumulated Q·K in attention (H38 tile half2, H47 GEMM QK16) | Both fail KLD at depth against the non-FA reference (H47: 0.618 vs tile's 0.581). Q·K must accumulate in fp32 |
| `GGML_CUDA_FORCE_MMQ`, `GGML_CUDA_F16` | No-ops on cc 6.0 (no dp4a; F16 compute already selected) |

### Measurement rules that constrain later phases

- **Attention levers: gate KLD against a non-FA reference** (`-sm layer -fa 0`, f16 KV; banked at `/root/h28-kld/h46-nofa-base.kld`) and require "no worse than tile". The tile-vs-lever KLD is dominated by a few hypersensitive prompt tokens (~30 KLD each) that any change to attention arithmetic flips. An absolute ≤ 0.001 bar can't be met even by tile vs non-FA (H46).

- Quote every prefill figure with its length **and ubatch**; every acceptance
  figure with its depth. Neither transfers.
- Speculative runs: ≥400 output tokens, **≥3 reps for every hypothesis arm** (user, 2026-10-15: rules out outliers).
  REPS=10 only for the final headline report (`scripts/run-headline-r10.sh`), once the lever work is done. Never compare acceptance
  across engines without pinning `n_max`/`p_min`.
- Phase 8 freezes the serve command before its first arm and moves one lever
  per run: the model file in Sweep Q, the drafter in Sweep D.
