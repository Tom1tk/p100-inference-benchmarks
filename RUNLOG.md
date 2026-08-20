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
