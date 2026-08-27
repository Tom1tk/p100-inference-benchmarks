# CLAUDE.md

Benchmark repo for serving Qwen3.8-27B on a dual Tesla P100 rig.
**Read [RUNBOOK.md](RUNBOOK.md) before running anything.** This file only
restates the rules that must never be re-derived from context.

## Hard limits

1. **Never let a card exceed 83 °C.** Abort the run if it does.
2. **Every run costs the user real electricity.** Before launching a sweep:
   - Prefer many short runs to few long ones.
   - Never re-measure a settled lever. Audit every arm against `results/` first
     and delete the ones that re-answer a closed question; cite the banked
     result as the control instead of re-taking it.
   - Abort early. If the conclusion is clear partway through, cancel the rest —
     and build that skip condition into the driver script so an unattended run
     stops itself.
   - Full detail and the worked example: RUNBOOK §2.1.
3. **Commit and push after every run, pass or fail.** A failed run is data.
4. **Don't change two variables at once.** One lever per run.

## Scope

- **PFlash is closed** — product, fork and technique. Do not reopen it.
- **DFlash2 v1 is excluded**; DFlash2 (v2) is in scope.
- Power cap is approved to **220 W per card, not 250 W**, and is gated on the
  user's own in-person PSU plug-meter check. Do not raise it unprompted.
- The single-GPU arm is **`Qwen3.8-27B-UD-IQ3_S`** — the same model at a lower
  quant, never a different model.

## Gotchas

- **Never edit a `.sh` file while a background job is executing it.** Bash reads
  scripts incrementally; the run corrupts and the file still passes `bash -n`.
- Model loads take 4–8 minutes cold with no output. That is not a hang.
- Read the results CSV for progress, not the driver's stdout — the sweep script
  writes rows incrementally, but its summary block only prints at the end.
