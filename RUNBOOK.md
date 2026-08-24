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
   is the machine-readable source of truth. Don't hand-edit it.
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
| `LLAMA_SPLIT_MODE_TENSOR not implemented for architecture 'qwen35'` | `-sm tensor` is arch-gated and `qwen35` isn't on the list — in any engine | Not a bug and not fixable here. Usable split modes for this model are `none`/`layer`/`row`, plus `graph` on `ik` |
| Abort at `fattn.cu:348` with turbo KV types | Known dispatch bug — `(turbo*, F16)` unsupported, only the reverse | Use symmetric `turbo3/turbo3` or `turbo3/q8_0`. See H7 |
| No `pp0` row in output | `-p 0` is a no-op prefill and is silently omitted | Expected, not a bug |
| Load appears hung, GPUs idle | Normal model load, or disk contention | See §2 timing table |
| `pi` hangs at startup, no request reaches the server | `pi` blocks on startup network operations in this environment | `PI_OFFLINE=1` — `run-web-bench.sh` sets it. Verified, not guesswork |
| Web-bench stage exits 124 | Hit `STAGE_TIMEOUT` (default 3600s) | Legitimate result — record it. Raise the env var only if the config is known-slow |
| Web-bench: `port N already in use` | Index reused, or a previous site is still hosted (by design) | Pick an unused index; check WEB_BENCH.md §5 |
| Server OOMs on load in Phase 5 | `CTX=32768` leaves too little VRAM beside a 21.5 GiB model | `CTX=16384 ./scripts/run-web-bench.sh ...` and note it in RUNLOG.md |
| Tool calls malformed / ignored in Phase 5 | `--jinja` missing, or a genuinely bad drafter | The script passes `--jinja`. If it's there, this is a **result** — see H8 |
| DFlash2 run shows poor acceptance instead of an error | A **v1** engine (`buun`/`ik`) silently ran the v1 path — same `draft-dflash` flag, no selector | Use `mainline`. Confirm the selector loaded in `logs/<label>.server.log` before recording. See H8 |
| `failed to load model` on `-sm none` | Target too big for one 16 GiB card | Only `UD-IQ3_S` (~11 GiB) fits a single P100 with usable KV |

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

### Phase 2 — MTP on/off
Winning engine(s) from Phase 1 × `UD-Q6_K_M` × MTP on vs off.

Requires the server, not `llama-bench` — the latter doesn't drive speculative
decoding. Per-engine flags are in METHODOLOGY.md and **differ meaningfully**
(`buun` uses `draft-mtp`, not `mtp`). Record acceptance rate, not just t/s.

**Done when:** H1 has a verdict with an acceptance rate to support it.

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

1. `GGML_CUDA_P2P=1` vs unset on the current working `-sm layer` config. Peer
   access is **off by default** in mainline and these cards can peer.
2. `GGML_CUDA_ALLREDUCE` three-way (`nccl` default / `internal` / `none`) on
   `-sm row`, which is the mode that actually exercises AllReduce.
3. Drafter placement: default vs `-devd CUDA0` vs `-devd CUDA1`.
4. `ik` graph knobs (`-smf16`, `-gap`, `-smgs`, `-sas`) — **after** the
   `-DGGML_NCCL=OFF` rebuild unblocks `-sm graph`.

Record per-GPU utilization from the temps log alongside throughput. These
hypotheses are about transfer stalls, so utilization is the evidence.

**Trap:** any `-ot` override silently disables pipeline parallelism. Compare
`-ot` runs against a no-`-ot` control, never against the Phase 1 baseline.

### Deferred
Context depths beyond 16k; power-limit experiment (175W vs 200W+ — log as a
separate row, don't silently change the cap mid-matrix); fine-grained
row-vs-layer comparison.
