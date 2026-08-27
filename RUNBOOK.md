# Runbook — operating procedure

This is the procedure for executing benchmark runs on this rig. Follow it in
order. It assumes no prior context beyond what's in this repo.

---

## 0. Non-negotiables

1. **Never let a card exceed 83°C.** Abort the run if it does. Hard shutdown
   is 87°C and these are passively cooled cards behind a custom fan shroud.
2. **Commit and push after every test run** — pass or fail. See §6. A failed
   run is data; losing it wastes 5–20 minutes of load time to rediscover.
3. **Don't change two variables at once.** One engine, one quant, one split
   mode per run. The whole point is attributable comparison.
4. **Don't silently change fixed parameters** in [METHODOLOGY.md](METHODOLOGY.md).
   If a run needs different parameters, record why in [RUNLOG.md](RUNLOG.md).

---

## 1. Preflight

Run before the first benchmark of any session:

```bash
nvidia-smi -L                       # expect exactly 2× Tesla P100-PCIE-16GB
nvidia-smi --query-gpu=index,temperature.gpu,power.limit,persistence_mode \
           --format=csv             # expect ~45-50°C idle, 175W cap, persistence Enabled
df -h /root                         # models are 15-22GB each; check headroom
```

Expected baseline: idle 45–50°C, power limit 175.00 W on both cards,
persistence mode Enabled.

If persistence or the power cap is missing (they don't survive a host reboot):

```bash
nvidia-smi -pm 1
nvidia-smi -pl 175 -i 0 && nvidia-smi -pl 175 -i 1
```

**Check for competing disk I/O.** Model loads are disk-bound. If a download or
copy is running, loads take 2–3× longer. Either wait or note it in the run log
— it does not affect GPU-phase numbers, only wall-clock.

---

## 2. Running one benchmark

```bash
cd /root/p100-benchmarks
./scripts/run-bench.sh <engine> <model-path> <split-mode> <label> [extra args...]
```

- `<engine>` — `pflash` | `buun` | `ik` | `mainline`
- `<split-mode>` — see the support matrix in [METHODOLOGY.md](METHODOLOGY.md).
  **They differ per engine.** `ik` has `graph` but not `row`/`tensor`;
  `pflash`/`buun` have `row`/`tensor` but not `graph`.
- `<label>` — used for filenames and the commit message. Use
  `phase<N>-<engine>-<splitmode>-<quant>`, e.g. `phase1-buun-layer-q6k`.

Example:

```bash
./scripts/run-bench.sh ik /root/Qwen3.8-27B-UD-Q6_K_M.gguf graph \
  phase1-ik-graph-q6k -cuda graphs=0
```

The script does preflight temp checks, starts telemetry, runs the benchmark,
stops telemetry, writes logs and CSV, then commits and pushes.

### Expected timing — read this before assuming a hang

| Phase | Duration | What it looks like |
|---|---|---|
| Model load | **4–8 min** (longer under disk contention) | Process in `D` state, GPUs at 0% and idle temps, no output past the CUDA device banner |
| Prefill sweep | 3–6 min | GPUs alternating to 100%, temps climbing to ~60°C |
| tg128 | ~30 s | One GPU at a time under `layer` split |

**A 21.5GB model takes minutes to load and produces no output while it does.**
This is the single most common false alarm. Confirm progress rather than
killing it:

```bash
cat /proc/<pid>/io      # read_bytes should be climbing toward the file size
```

### Don't poll with sleep loops

Launch long runs in the background and wait for completion notification rather
than chaining sleeps. `run-bench.sh` is designed to be launched in the
background and to be self-contained.

---

## 3. Thermal safety

`run-bench.sh` starts `scripts/gpu-monitor.sh`, which samples every 5s into
`logs/<label>.temps.log` (format: `HH:MM:SS idx, tempC, watts, util%` per card).

Watch a live run:

```bash
tail -f logs/<label>.temps.log
```

Thresholds:

| Temp | Action |
|---|---|
| ≤ 80°C | Normal. Observed peak so far is 63°C under full load. |
| 80–82°C | Warn, finish the current run, don't start another until cooled |
| ≥ 83°C | **Abort immediately** (`pkill -f llama-bench`), record in RUNLOG.md |
| 87°C | Hardware shutdown — must never be reached |

Brief power readings slightly above 175W are normal sampling overshoot; the
cap is enforced on a rolling average. Sustained draw well above it is not
normal and is worth investigating.

---

## 4. Recording results

After each run:

1. **`results/all-results.csv`** — appended automatically by the script. This
   is the machine-readable aggregate. Don't hand-edit it.

   > **Do not parse this file by column position or by `csv.DictReader`.**
   > Different engines emit different llama-bench schemas — pflash 43 columns,
   > buun 46, ik 56 — and they are all appended under the single header written
   > by whichever engine ran first. `avg_ts` therefore sits at a different index
   > per engine, and a positional or header-keyed read silently returns the
   > wrong column (observed: ik decode read as 0.02 t/s instead of 22.05).
   > **Parse `results/raw/<label>.csv` instead** — each carries its own correct
   > header. Use `all-results.csv` only for grepping which labels exist.
2. **[RESULTS.md](RESULTS.md)** — add a curated row to the relevant phase
   table. Include split mode, peak temp, and anything anomalous.
3. **[RUNLOG.md](RUNLOG.md)** — add an entry only if something notable
   happened (a crash, a parameter deviation, a surprising number). Routine
   successful runs don't need narrative.
4. **[HYPOTHESES.md](HYPOTHESES.md)** — if the run bears on H1–H9, update that
   hypothesis's status. This is easy to forget and is the main reason to run
   these benchmarks at all.

For Phase 5 web-bench runs there are two extra steps: claim the index in
[WEB_BENCH.md](WEB_BENCH.md) §5's port registry, and score the generated site
by hand against the quality checklist in WEB_BENCH.md §4. Throughput alone does
not decide that phase.

---

## 5. When a run fails

Do not just retry. Failures are results.

1. Capture the error — `logs/<label>.log` holds full stdout+stderr including
   any stack trace.
2. Record it in [RUNLOG.md](RUNLOG.md) with the exact error string and the
   command that produced it.
3. Check whether it bears on a hypothesis — the NCCL crash resolved most of H5
   before a single timing number was collected.
4. **Commit and push the failure** before attempting a fix.
5. Only then diagnose. If the fix is a rebuild, record the new commit hash and
   build flags in [METHODOLOGY.md](METHODOLOGY.md).

### Known failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `ncclAllReduce failed with status 1`, abort at `reduce.cu:169` | `ik_llama` built with NCCL (`GGML_NCCL` defaults **ON**); no NVLink on this rig | Rebuild ik_llama with `-DGGML_NCCL=OFF` |
| `error: invalid parameter for argument: -sm` | Split mode not supported by that engine | Check the per-engine matrix in METHODOLOGY.md |
| `ggml-cuda.cu:106: CUDA error` on mainline `-sm tensor` | NCCL AllReduce, the Linux default, is broken on this rig — same root cause as the `ik` row above. The line number is the generic abort dispatcher; get the real cause from the backtrace frame `ggml_backend_cuda_comm_allreduce_nccl` | Set `GGML_CUDA_ALLREDUCE=none`. (`internal` also works but only because it falls back to the same path — see below.) No rebuild needed |
| `LLAMA_SPLIT_MODE_TENSOR not implemented for architecture '<arch>'` | `-sm tensor` is arch-gated — but the gate is a **denylist** (`default: return true`), so most archs including `qwen35` are permitted. Earlier guidance here claimed the reverse and was wrong | If you see this for `qwen35`, something else changed — check `llm_arch_supports_sm_tensor()` in `src/llama-arch.cpp` |
| Abort at `fattn.cu:348` with turbo KV types | Known dispatch bug — `(turbo*, F16)` unsupported, only the reverse. Turbo KV types exist only in `buun` and `pflash` | Use symmetric `turbo3/turbo3` or `turbo3/q8_0`. See H7 |
| No `pp0` row in output | `-p 0` is a no-op prefill and is silently omitted | Expected, not a bug |
| Load appears hung, GPUs idle | Normal model load, or disk contention | See §2 timing table |
| `pi` hangs at startup, no request reaches the server | `pi` blocks on startup network operations in this environment | `PI_OFFLINE=1` — `run-web-bench.sh` sets it. Verified, not guesswork |
| Web-bench stage exits 124 | Hit `STAGE_TIMEOUT` (default 3600s) | Legitimate result — record it. Raise the env var only if the config is known-slow |
| Web-bench: `port N already in use` | Index reused, or a previous site is still hosted (by design) | Pick an unused index; check WEB_BENCH.md §5 |
| Server OOMs on load in Phase 5 | `CTX=32768` leaves too little VRAM beside a 21.5 GiB model | `CTX=16384 ./scripts/run-web-bench.sh ...` and note it in RUNLOG.md |
| Tool calls malformed / ignored in Phase 5 | `--jinja` missing, or a genuinely bad drafter | The script passes `--jinja`. If it's there, this is a **result** — see H8 |
| DFlash2 run shows poor acceptance instead of an error | A **v1** engine (`buun`/`ik`) silently ran the v1 path — same `draft-dflash` flag, no selector | Use `mainline`. Confirm the selector loaded in `logs/<label>.server.log` before recording. See H8 |
| `failed to load model` on `-sm none` | Target too big for one 16 GiB card | Only `UD-IQ3_S` (~11 GiB) fits a single P100 with usable KV |
| Running script dies with a syntax error partway through (`break: only meaningful in a loop`, `<message>: command not found`) | **A script was edited while a background job was executing it.** Bash reads a script incrementally, so an edit shifts the byte offsets under the running shell and it resumes mid-token. The file is fine afterwards — `bash -n` passes — which makes this look like a phantom | Never edit `scripts/*.sh` while `pgrep -f llama-server\|llama-bench` shows work in flight. Re-run the affected cell; discard nothing else |
| `ggml-backend-meta.cpp:543: GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0)` on DFlash2 | DFlash2 borrows the target's `output.weight` via `ctx_other`; `-sm tensor` shards it on axis 0 and the split planner can't handle a per-row op over it | Not fixable by configuration. DFlash2 is layer-split-only. See H8 |
| `internal AllReduce init failed (n_devices != 2?)` | Misleading message — the real gate is compute capability. The internal pipeline needs `__nanosleep` (sm70+); P100 is sm_60 | Expected on this rig, harmless. Use `GGML_CUDA_ALLREDUCE=none`; `internal` reaches the same butterfly path. See H10 |

---

## 6. Git protocol — commit and push every run

**This is mandatory and applies to failed runs too.**

`run-bench.sh` does this automatically at the end of each run. If you run a
benchmark by hand, or edit any document, do it manually:

```bash
cd /root/p100-benchmarks
git add -A
git commit -m "bench: <label> — <one-line outcome>"
git push
```

Commit message conventions:

| Prefix | Use for |
|---|---|
| `bench:` | A benchmark run (pass or fail) |
| `docs:` | Documentation changes with no new data |
| `build:` | Engine rebuilds, flag changes |
| `fix:` | Corrections to previously recorded data or claims |

Remote: `https://github.com/Tom1tk/p100-inference-benchmarks.git` (private).

If a push fails, do not discard the commit — the local history is the record.
Retry, and note the failure in RUNLOG.md if it persists.

---

## 7. Phases and completion criteria

Run phases in order. A phase is done when every cell has either a number or a
recorded reason it couldn't produce one.

### Phase 1 — engine baseline
No MTP, no TurboQuant. Establishes the apples-to-apples comparison.

- 3 engines × `UD-Q6_K_M` × {`none` (single-GPU reference), `layer`}
- The `none` leg needs `UD-IQ3_S`, not `Q6_K_M` — 21.5 GiB does not fit one
  card. Record it as a separate quant row, not as a `Q6_K_M` single-GPU number
- Plus `graph` for `ik` only — **blocked until the NCCL rebuild**
- Full prefill depth list + tg128, `-r 3`

**Done when:** every engine has `none` and `layer` numbers, and the
single-vs-dual comparison is recorded. Answers part of H6.

**Status (2026-08-25): complete except the `none` leg**, which the operator
deferred. It needs `UD-IQ3_S` — the only target that fits one 16 GiB card.
Everything else is in RESULTS.md, on `UD-Q4_K_M` rather than `Q6_K_M`.
Selected configuration: **`mainline-rebased` · `-sm tensor` ·
`GGML_CUDA_ALLREDUCE=none`**.

### Phase 2 — MTP on/off
Winning engine(s) from Phase 1 × `UD-Q6_K_M` × MTP on vs off.

Requires the server, not `llama-bench` — the latter doesn't drive speculative
decoding. Per-engine flags are in METHODOLOGY.md and **differ meaningfully**
(`buun` uses `draft-mtp`, not `mtp`). Record acceptance rate, not just t/s.

**Done when:** H1 has a verdict with an acceptance rate to support it.

**Status (2026-08-25): complete.** H1 has a verdict — refuted as stated, and
the speedup is depth-dependent (1.05× at 4k, 1.48× at 16k) rather than a single
number. Best configuration: **`-sm tensor` + MTP-Q4_0 + `f16` KV, 29.26 t/s at
16k depth.**

Two measurement rules this phase established, both learned the hard way:

- **Never size a speculative-decoding run at 64 tokens.** A short generation
  overstated the MTP speedup by 13x (90.0% acceptance vs 40.1% at 400 tokens).
  Use `N_PREDICT=400` minimum, and quote depth alongside every acceptance rate.
- **Acceptance rate is not comparable across engines** unless `n_max`/`p_min`
  are pinned on both. `buun` reported 82.4% where `mainline-rebased` reported
  73.3% on an identical drafter, prompt, and budget — different defaults, not a
  better drafter. Compare end-to-end decode t/s instead.

### Phase 3 — quant sweep
Winning engine/split from Phases 1–2 × `IQ3_S` / `Q4_K_M` / `Q5_K_M` / `Q6_K_M`
× tg128 + pp16384 only. Not the full depth sweep — this phase is about quant
choice, not re-confirming depth scaling.

Run `IQ3_S` **twice**: once on `-sm layer` for the quant-curve row, and once on
`-sm none` (single card, `CUDA_VISIBLE_DEVICES=0`) for the single-GPU reference
that H6 needs and that every other quant is too large to provide. Expect its
prefill to trail the K-quants — IQ dequantization is compute-heavy and sm_60 has
no dp4a — so read it as the cost of single-GPU, not as a like-for-like quant
comparison.

### Phase 4 — targeted hypothesis tests
H2 (XL split-stall), H3 (stock quant quality), H4 (TurboQuant KV), H5 (NCCL),
H7 (dispatch bug). Each is a small targeted run, not a matrix pass. See
[HYPOTHESES.md](HYPOTHESES.md) for the specific test each needs.

### Phase 5 — agentic web-build benchmark
The final phase, and the only one that measures *usability* rather than
throughput. Full procedure in **[WEB_BENCH.md](WEB_BENCH.md)** — read it before
running, the port scheme and quality scoring are not obvious from the script.

```bash
./scripts/run-web-bench.sh <engine> <model-path> <split-mode> <label> <index> [server args...]
```

Matrix: the winning engine/split from Phases 1–3 × each **drafter arm** —
none (control), MTP, DFlash2-Q4, DFlash2-Q8. All four are runnable as of
2026-08-24; the DFlash2 arms require `engine = mainline` (see H8).

Because `mainline` is a different engine, run its **own** no-drafter control as
an arm. Comparing DFlash2-on-mainline against MTP-on-buun would confound drafter
with engine, which is the one thing this repo exists to avoid.

Each run claims an index and leaves its site hosted at `4000 + index`, so at
the end every model version is browsable side by side.

**Done when:** every drafter arm has a total task time, token count, and
prefill/decode avg/min/max, plus a hand-scored quality verdict — or a recorded
reason it couldn't produce one. Answers H9 and the practical half of H1/H8.

### Phase 6 — dual-GPU transport and placement (H10–H12)
Added 2026-08-24 after a survey found several untouched inter-GPU controls.
These are cheap, mostly runtime-switchable, and apply to whichever config wins
Phases 1–3. Full knob list in METHODOLOGY §3.

Highest value first:

1. `GGML_CUDA_P2P=1` vs unset on **`-sm tensor`**. Peer access is **off by
   default** in mainline and these cards peer in both directions. Test it on
   tensor split, not layer — tensor moves far more cross-GPU data. This is the
   best remaining cheap test in the project.
2. ~~`GGML_CUDA_ALLREDUCE` three-way~~ — **done**. `nccl` aborts; `internal`
   and `none` are identical to within 0.2%. Correctness switch, not a
   performance one. See H10.
3. ~~Drafter placement~~ — **done**, refuted at 4k and 16k. See H11.
4. `ik` graph knobs (`-smf16`, `-gap`, `-smgs`, `-sas`) — unblocked by the
   `build-nonccl` rebuild, but lower priority now that `-sm tensor` beats
   `-sm graph` on prefill at every depth.

Record per-GPU utilization from the temps log alongside throughput. These
hypotheses are about transfer stalls, so utilization is the evidence.

**Trap:** any `-ot` override silently disables pipeline parallelism. Compare
`-ot` runs against a no-`-ot` control, never against the Phase 1 baseline.

### Phase 7 — prefill and TTFT (H13–H22)

**Two levers remain after 2026-08-25**, once TurboPrefill (H16) and PFlash (H21)
were withdrawn: the **power cap** (H15, cell 8 — requires the user's careful manual
involvement at every step) and the **chunked GDN path** (H22, cell 7). Everything
else in this phase is measurement, or a work-reducing lever rather than a way to
make the existing kernels faster.

**The current priority.** Opened 2026-08-25 from
[Research/prefill-ttft-2026-08-25.md](Research/prefill-ttft-2026-08-25.md) —
read it before running anything here; it closes several obvious-looking knobs at
the source level and establishes the hardware floor that bounds every result.

Run in this order. The first two are gates: they decide whether the rest of the
phase is aimed at the right target.

| # | Cell | Hypothesis | Why this order |
|---|---|---|---|
| 1 | `-p 16384,32768,65536,100000 -n 0`, tensor, f16 KV, **`-b 2048 -ub 2048`** | H13 | We have no data past 16k, and no data at all at the new ubatch. **Highest-value run in the project** |
| 2 | `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` vs `f16` at 16k, **at `-ub 2048`** | H18 | One run. Must be re-based on the new ubatch — running it at `-ub 512` measures the wrong regime |
| ~~3~~ | ~~`-ub` sweep~~ | H14 | **DONE 2026-08-25 — CONFIRMED, +63%.** `-ub 2048` is the new default; `-ub 8192` OOMs. Re-baseline everything else on it |
| 4 | `--cache-ram 20000 --cache-idle-slots` on WEB_BENCH multi-turn | H19 | Highest practical value for the actual use case; costs a flag |
| 5 | sm_60 FP16 fast-path patch, rebuild, one Phase 2 cell + one NIAH tier | H17 | Free by hypothesis; also un-confounds buun-vs-rebased quality |
| 6 | **27B-IQ3_S single-GPU** vs two-card 27B-Q4_K_M — depth ceiling, then speed, then NIAH | H20 | Harness and fixtures exist. **Reframed 2026-08-26: the fallback is the same model at a lower quant, never a different model.** Needs a two-card Q4_K_M NIAH control or the quality leg measures nothing |
| 7 | **Chunked GDN path** — patch `fused_gdn_ch=false` behind an env var, rebuild, A/B at `-ub 2048` | H22 | The last kernel-level lever. ~2-line patch, not a kernel project. **Run cell 2 first** — H18 sizes the prize. Check the load log for CPU-fallback warnings before trusting any number |
| 8 | Power cap 175→200→220 W | H15 | **Approved to 220 W**, but blocked on the user's in-person PSU plug-meter check. Thermally the riskiest cell — temperature log is the primary output |
| ~~9~~ | ~~PFlash-style selection spike~~ | ~~H21~~ | **Withdrawn 2026-08-25.** PFlash is sm_80-only with no v2; hand-porting it is not worth the build cost |
| ~~—~~ | ~~TurboPrefill build~~ | ~~H16~~ | **Withdrawn 2026-08-25** — user call. Forces `-sm layer`, costing 35% of decode to buy prefill; wrong trade against the four simultaneous targets |

**`-ub 2048` is now mandatory in every Phase 7 cell.** H14 found the old default
`-ub 512` to be a local minimum of the throughput curve, worth 63% less than 2048.
Any cell run at the old default measures the wrong regime and will have to be
repeated. Use `scripts/run-ubatch-sweep.sh` (env-overridable: `LABEL`, `BATCH`,
`UBATCHES`, `PROMPTS`, `REPS`) for further sweeps; note `-ub` is capped by `-b`,
so raising ubatch above 2048 requires raising `-b` too.

**Thermal note, and it is new.** A 100k prefill is 8–16 minutes of *sustained*
compute — several times longer than any run this repo has done, and prefill loads
the cards harder than decode. The 83 °C limit has never been tested under this
profile. Monitor cell 1 actively; do not background it and walk away.

**Reuse what exists.** `/root/niah_test/` has a working harness and generated
fixtures at 8k/32k/64k/100k (single and multi-needle). `pflash-llama.cpp` ships a
built `llama-niah` with fixtures to 128k — that binary and its fixtures are the
only reason the fork is still on disk. H20 is mostly a re-run, not new harness
work. Its July baseline is a **Qwen3.5-9B** run, so it validates the harness but
is **not** a quality baseline for anything in the current plan: every model in
Phase 7 is Qwen3.8-27B, and the single-GPU arm is `Qwen3.8-27B-UD-IQ3_S`
(11.2 GiB, on disk). Any quality comparison needs its own 27B control.

### Deferred
~~Context depths beyond 16k~~ — **now Phase 7 cell 1, the top priority**.
~~Power-limit experiment~~ — **approved 2026-08-25 to a 220 W ceiling** (not 250 W),
now Phase 7 cell 8. Still blocked on one thing: the user must measure PSU wall draw
with a plug-socket meter in person before the first run above 175 W. Log the cap as
a separate column; don't change it silently mid-matrix. Remaining deferred:
fine-grained row-vs-layer comparison; Phase 1's single-GPU (`none`) leg on `UD-IQ3_S`.

### Closed — do not spend runs here

| Line of enquiry | Why it's closed |
|---|---|
| TurboQuant KV | −14.3% decode for 788 MiB. H4/H7 |
| `buun` as an engine | Behind `mainline-rebased` on prefill (12–14%) and on decode with MTP (3.9%); its one exclusive feature is TurboQuant. H6 |
| DFlash2 | Layer-split-only, and `tensor + MTP` beats `layer + DFlash2` by +71% decode. H8 |
| `GGML_CUDA_ALLREDUCE` tuning | Only butterfly runs on Pascal; `internal` and `none` are the same path. Use `none`. H10 |
| Drafter placement (`-devd`) | Moves VRAM, not throughput. H11 |
| `ik` `-sm graph` knobs (H12) | Low value now — `-sm tensor` beats graph on prefill at every depth and is within 8% on decode, and `ik` has no path to tensor split |
| **PFlash — product, fork and technique** | Requires **sm_80+** — the `mean_K → score → select → sparse_fwd` kernels plus BSA target Ampere, and no v2 exists, so it is architecturally unavailable. H21 briefly reopened the *technique* as a hand-port; **withdrawn 2026-08-25** — not worth the build cost with no upstream to track. Note also that the earlier "too lossy" quality verdict was formed on a **7900 XTX, not on these P100s**, so the sm_60 arithmetic bug (H17) never confounded it. `pflash-llama.cpp` stays on disk only for its built `llama-niah` binary and fixtures |
| **vLLM / vllm-pascal / pascal-pkgs-ci** | vLLM needs cc ≥ 7.0; `vllm-pascal` is discontinued, its successor tops out at vLLM **0.10.0** and is "soft-broken due to PyTorch". 0.10 long predates `qwen35` hybrid-SSM support |
| **SGLang, TensorRT-LLM, ExLlamaV3, ktransformers** | sm_75 / sm_80+ minimum |
| **MLC-LLM / TVM** | No `qwen35` hybrid support; we would be writing GDN kernels in TVM with worse tooling, and MLC prefill trails llama.cpp where measured |
| **tinygrad** | Not an inference engine — no GGUF loader, no gated-delta-net op, no hybrid-SSM support, no quantised Pascal kernels, no TP serving. Parity means rewriting the stack we already have |
| **FlashQLA** | Hopper SM90 + TMA + warpgroup MMA only |
| `GGML_CUDA_FORCE_MMQ` | `__dp4a` needs cc 6.1; P100 is 6.0. `mmq.cu:316` returns false unconditionally. Quantised prefill always goes dequant → cuBLAS |
| `GGML_CUDA_F16` | Already effectively on — `ggml-cuda.cu:1626` picks `CUBLAS_COMPUTE_16F` for cc 600, and `prefer_f32_output` is set only for cc == 700 |
| KV quant to reach 100k | Unnecessary: f16 KV at 100k is 6.25 GiB, and 100k fits VRAM. KV quant doesn't affect prefill at all |

### Unmeasured — do not quote

**`tensor + MTP + GGML_CUDA_P2P=1` was never measured.** P2P (+2.9% decode) was
measured drafter-free on `llama-bench`; MTP (29.26 t/s) on `llama-server` at
depth. The combined cell was queued twice and ran neither time. The defensible
best-measured number is **29.26 t/s**, without P2P.

---

## What this project is for

The deliverable is **a server launch command**, not a table. Read every phase as
"does this change what goes in that command." Four targets must hold *at once* —
100k context, highest decode, highest prefill, and output quality indistinguishable
from baseline — and almost every lever trades one against another. The quality
target is the binding constraint and has the least data behind it; a throughput win
from a lossy technique does not count until it clears a NIAH gate.

See [README.md](README.md#the-objective) for the full statement and the lever map.
