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
