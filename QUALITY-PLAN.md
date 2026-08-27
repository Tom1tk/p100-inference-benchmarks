# Phase 8 — the quality gate

**Plan only. Nothing here has been run.** Written 2026-08-27, after H26 closed
the speed and context legs at 100k. Quality is the fourth objective and the only
one with no data at any depth.

**Deliberately last.** Every other lever is being dialled in first, because none
of them change *what* the model outputs but several of them change *how long it
takes to find out*. The arithmetic for that is in §1 and it is the whole reason
this document exists before the runs do.

---

## 1. What the sweep actually costs, and which levers move it

At the H26 serve config a 100k prefill takes **482 s (8.0 min)** at 207.4 t/s.
A NIAH-style answer is ~32 tokens, which at 20.02 t/s is **1.6 s**.

> **Cost ≈ (number of distinct 100k prefills × 8 min) + (model loads × 3 min).**
> Generation is 0.3% of it.

That reframes one thing worth saying plainly: **decode speed is almost
irrelevant to the cost of the quality sweep.** Prefill rate matters, and the
*number of prefills* matters far more than either. So the prerequisites are
ranked below by their effect on that count, not by how interesting they are.

| # | Prerequisite | Effect on the sweep | Why it must come first |
|---|---|---|---|
| 1 | **H19 — prompt-cache reuse** (`--cache-ram`, `--cache-idle-slots`, `-ctxcp`) | **~10x.** Turns *N prefills per model* into *one* | If the 100k prefix is cached, question 2..N cost seconds instead of 8 min. This is the single largest cost lever in the plan and it is already an open hypothesis that costs nothing but flags |
| 2 | **Fixture design for prefix reuse** (§2.2) | Multiplies #1 by making it usable | Free — it is a decision, not a run. But if fixtures vary the *middle* of the haystack, cache reuse buys nothing, so it has to be settled before any fixture is generated |
| 3 | **H17 — sm_60 FP16 fast-path fix** | ~0 on cost, **fatal on validity** | It changes model numerics. Landing it *after* the sweep invalidates every number in the sweep. Settle it either way — applied or rejected — before the first quality run |
| 4 | **Sweep D, the drafter equivalence check** (§3) | Licenses dropping the drafter for all of Phase 8 | Frees ~0.65 GiB/card, removes a variable, and may be what makes the Q6 baseline fit at all. Costs ~45 min and also settles the open drafter re-rank |
| 5 | **H15 — power cap to 220 W** | Raises prefill, so shortens every arm | Worth having before, not worth blocking on. Gated on the user's in-person PSU plug-meter check |
| 6 | Engine, `-sm`, `-ub`, `-b`, KV, ALLREDUCE | Already settled | No work. Do not re-open mid-sweep |

**Concretely:** Sweep Q is **~2 hours with prefix reuse and ~14 hours without**.
That gap is the justification for doing everything else first.

**Freeze point.** Before the first quality run, write the exact serve command
into this file and change nothing in it for the duration. One lever per run
applies to Phase 8 too: the lever is *the model file* in Sweep Q and *the
drafter* in Sweep D, and nothing else moves.

---

## 2. Sweep Q — quality across quants

### 2.1 The baseline problem, settled by definition

"Indistinguishable from baseline" has never been given a referent in this repo,
and it needs one before any arm runs. An f16 27B reference is **impossible on
this rig** — 54 GB against 32 GB of VRAM and 28 GB of host RAM — so there is no
ground truth available at any price.

> **Assumption, stated because everything downstream rests on it:** the baseline
> is **the highest quant that loads at the serve config**. Every other quant is
> scored as agreement with that one, not against absolute truth.

This is weaker than a true reference and should be reported as such. It answers
the question that actually matters — *"can we drop to a cheaper quant without
losing anything we can detect?"* — and not the question of how far the whole
ladder sits from f16.

### 2.2 Step 0 — which quants load (load-only, no generation)

All six models are already on disk. Predicted VRAM per card at 100k / `q8_0` KV,
extrapolated from H26's measured 12,147 MiB for Q4_K_M (limit **15.89 GiB**):

| Model | On disk | Predicted MiB/card | Verdict |
|---|---|---|---|
| `UD-IQ3_S` | 11.2 GiB | ~10.0 GiB | fits easily |
| `UD-Q4_K_M` | 15.3 GiB | **11.86 GiB (measured)** | the H26 control |
| `UD-Q4_K_XL` | 16.4 GiB | ~12.4 GiB | should fit |
| `UD-Q5_K_M` | 18.4 GiB | ~13.4 GiB | should fit |
| `UD-Q6_K_M` | 21.5 GiB | ~14.9 GiB | **marginal — ~1 GiB of headroom** |
| `Uncensored-noMTP-Q4_K_M` (stock) | 15.4 GiB | ~11.9 GiB | fits; H3 only, confounded |

**Run it as a descending ladder, load-only, ~3 min per arm.** Start at Q6_K_M:
if it loads, it is the baseline and everything below it is guaranteed to load,
so the rest of step 0 is skipped entirely. If Q6 fails, drop the drafter (§3
licenses this, ~0.65 GiB/card) and retry once; if it still fails, Q5_K_M becomes
the baseline. **No tokens are generated in step 0** — it is a load check, and
planning quality arms for a config that cannot load is exactly the kind of run
rule zero exists to prevent.

### 2.3 The measure — two layers, because NIAH alone is not enough

**Layer 1 — multi-needle NIAH at 100k.** The retrieval floor. Reuses
`/root/niah_test/run_engine.py` and `gen_fixtures.py`, which are working and
were validated in July. Pass/fail per needle, scored by string match.

**Layer 2 — code comprehension on a real corpus.** `prompts/h26-100k-prompt.txt`
already exists: 97,620 tokens of real `.cpp`/`.cu`/`.h`/`.md`. Layer 1 tests
whether the model can *find* a planted fact; the deliverable is a **coding**
assistant, and finding is not understanding. Layer 2 asks questions about the
corpus that require following code across files.

**Scoring constraint, fixed now:** every question in both layers must have a
**short, mechanically checkable answer** — a function name, a number, a file
name, a yes/no. No judge model is available locally at sane cost, and hand-
scoring 100+ answers is not going to happen honestly. If a question cannot be
scored by `grep`, it does not go in the fixture.

### 2.4 Fixture shape — one haystack, many questions

This is the design decision that makes prerequisite #1 pay:

- **One fixed 100k haystack per layer.** Generated once, byte-stable, committed.
- **All needles present simultaneously**, planted at ~10%, 25%, 50%, 75%, 90%
  depth, so depth coverage costs no extra prefills.
- **Questions appended at the very end**, one request each, so every request
  shares the same ~100k prefix and only the last few hundred tokens differ.

The conventional NIAH design — move one needle, re-prefill — would cost one
prefill per position per model and is what turns a 2-hour sweep into a 14-hour
one. Depth coverage is bought with multiple needles instead of multiple runs.

**Known limitation, recorded rather than hidden:** needles that coexist can
interfere with each other in a way an isolated needle does not, and a fixed
haystack tests one document rather than a distribution. This measures *relative*
degradation across quants, which is the question being asked, and it should not
be quoted as an absolute NIAH score or compared to published NIAH numbers.

### 2.5 Arms and abort rules

Ordered so the sweep can stop early:

1. **Baseline** (highest loading quant) — establishes the reference answers.
2. **`UD-Q4_K_M`** — what we currently serve. *This is the arm that matters.*
   If it matches the baseline, the deliverable is already correct and arms 3-5
   are optional.
3. `UD-Q4_K_XL` — only if Q4_K_M shows a real gap and the extra 0.5 GiB/card
   might close it.
4. `UD-IQ3_S` — only if something below Q4 is wanted. H20/H25 closed IQ3 on
   *speed* grounds for single-GPU; on two cards it is a live option only if
   Q4_K_M disappoints.
5. `Uncensored-noMTP-Q4_K_M` — **H3, and confounded**: it differs from the UD
   model in fine-tune as well as quantizer, so it can suggest and never prove.
   Last, and only if there is appetite.

**Abort:** if the baseline and Q4_K_M agree on every needle and every layer-2
question, stop. The sweep has answered the deliverable's question and arms 3-5
cannot change the serve command.

---

## 3. Sweep D — drafters. This is an equivalence check, not a quality sweep

**The prediction is that this sweep has no quality content at all**, and it is
worth saying before spending power on it.

Speculative decoding is verified by the target model: every emitted token is
either a draft the target accepted or one the target produced itself. At
**temperature 0** the output is therefore a property of the target, and the
drafter can only change *how fast* it arrives. H24 already demonstrated this on
this rig — the 400 generated tokens were **byte-identical** across a change that
altered the drafter's own numerics, and identical across all reps.

So the right test is not a NIAH ladder across drafters. It is:

> **4 arms x 1 fixed 100k prompt x greedy (`temperature 0, top_k 1, seed 42`) x
> 400 tokens, then `sha256sum` the four outputs.**

Arms: `none`, `MTP-Q4_0`, `DFlash2-Q4_K_M`, `DFlash2-Q8_0` — all on disk.
Cost: ~4 x (3 min load + 8 min prefill) = **~45 min**.

**Two outcomes:**

- **All four hashes match** (expected). Drafters are quality-neutral, and this
  is settled permanently rather than re-litigated at each new config. They are
  then ranked on **speed alone**, which the same run measures — and that closes
  the open "drafter re-rank at `-ub 2048`/tensor/100k" gap, since every existing
  MTP-vs-DFlash2 comparison was taken at `-ub 512`/4k/`-sm layer` where DFlash2
  led. **One run, two answers.**
- **They differ.** That is a *bug*, not a quality finding — either in the
  verification path or in how the MTP head is integrated. It would then become a
  real investigation, and the drafter would be excluded from the serve command
  until it was understood. Escape hatch, not the expected path.

**Scope caveat:** this holds at temperature 0. Above it, speculative decoding is
distribution-preserving but not stream-identical, so byte-equality is the wrong
test and we have no cheap tooling for the right one. Serve coding workloads at
temperature 0 — which is the sane default anyway — and this caveat never binds.

**Sweep D runs before Sweep Q**, because its result is what licenses dropping
the drafter for the whole of Phase 8.

---

## 4. Order of operations

```
1. H17 settled (applied or rejected)          -> verify: one Phase 2 cell, throughput unchanged
2. H19 tuned                                   -> verify: turn-2 TTFT at 100k drops >90%
3. Sweep D (4 arms, hash compare)              -> verify: 4 identical sha256, + speed re-rank
4. Freeze the serve command into this file     -> verify: written down, one lever per run after
5. Sweep Q step 0 (descending load ladder)     -> verify: baseline quant identified, no tokens spent
6. Sweep Q layer 1 (NIAH, baseline + Q4_K_M)   -> verify: per-needle pass/fail table
7. Sweep Q layer 2 (code comprehension)        -> verify: per-question agreement table
8. Arms 3-5 only if 6/7 show a real gap        -> abort rule in 2.5
```

Steps 1-2 are prerequisites and are *not* quality work. Steps 3-8 are Phase 8
proper and total roughly **3 hours of GPU** if step 2 lands, against ~14 without.

## 5. What this plan does not cover

- **Absolute quality.** There is no f16 reference on this hardware. Everything
  here is relative to the highest quant that loads.
- **Agentic quality.** WEB_BENCH (Phase 5) is the harness for "does it build a
  working thing", and it is a different and much more expensive question.
- **Long-horizon degradation.** All of this is single-turn. Whether quality
  holds across a 20-turn agentic session with cache reuse is a Phase 5 question.
