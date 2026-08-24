# Phase 5 — agentic web-build benchmark

The final phase. Everything before it measures the engine in isolation with
`llama-bench`; this measures whether a model version is actually **usable** for
real one-shot agentic work.

`llama-bench` reports throughput on synthetic fixed-length prompts. It cannot
tell you whether a config completes a multi-step task, how it behaves once the
context fills with tool output, or whether speculative decoding still helps when
the workload is code and JSON rather than prose. That is what this phase is for.

---

## 1. What it does

For each model version, `scripts/run-web-bench.sh`:

1. Starts `llama-server` on that engine/model/split with `--jinja` (required —
   the agent drives the model through tool calls).
2. Starts a recording proxy between the agent and the server.
3. Runs `pi` through the three prompts in [prompts/web-bench.md](prompts/web-bench.md),
   one continuous session, in `sites/<label>/`.
4. Tears down, summarizes, commits, and pushes.

The prompts build an Express site about the model itself: a working page and
systemd service (stage 1), an elaborate restyle with a JS interactive element
(stage 2), and a canvas mini-game in a separate static file (stage 3). They are
written to be completed **one-shot** — a model that stops to ask the operator a
question has failed the stage, and that is a result worth recording.

Stage 3 in particular is the stress test the earlier phases can't provide: a
model that produces plausible-looking but broken game logic, or that mangles the
tool calls needed to write `public/game.js`, will look perfectly healthy in a
`llama-bench` t/s number.

---

## 2. Running one

```bash
cd /root/p100-benchmarks
./scripts/run-web-bench.sh <engine> <model-path> <split-mode> <label> <index> [extra llama-server args...]
```

```bash
# baseline, no speculative decoding
./scripts/run-web-bench.sh buun /root/Qwen3.8-27B-UD-Q6_K_M.gguf layer p5-buun-layer-q6k 0

# same config with MTP, for the usable-speed delta
./scripts/run-web-bench.sh buun /root/Qwen3.8-27B-UD-Q6_K_M.gguf layer p5-buun-layer-q6k-mtp 1 \
  -md /root/mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 3
```

`<index>` must be unique per model version and is the whole port scheme:

| Resource | Port | Example (index 3) |
|---|---|---|
| Website | `4000 + index` | 4003 |
| `llama-server` | `8100 + index` | 8103 |
| Metrics proxy | `8200 + index` | 8203 |

**Sites are left running on purpose.** Each run creates `<label>.service`, so
after the phase every model's site is browsable side by side at 4000, 4001,
4002… That is the point of the offset — the original single-port prompt could
only ever host one at a time. The script refuses to start if any of its three
ports is already bound, which is what catches a reused index.

Keep the registry in §5 current so indices don't get reused.

---

## 3. What gets measured

Per request, captured by the proxy from the `timings` object that all three
engines return:

| Field | Meaning |
|---|---|
| `prompt_n`, `prompt_per_second` | prefill tokens and prefill t/s |
| `predicted_n`, `predicted_per_second` | generated tokens and decode t/s |

Aggregated per stage and for the whole task into:

- **Total task time** — wall clock, all three stages, including tool execution
  and `npm install`. This is the number that answers "is this usable?"
- **Total tokens generated** — sum of `predicted_n`.
- **Prefill t/s** — avg / min / max across requests.
- **Decode t/s** — avg / min / max across requests.

Avg/min/max matter more than the mean alone: decode t/s degrades as the agent's
context fills, so a wide min–max spread means the model slows down exactly when
the task gets hard. A `llama-bench` tg128 number never shows this.

Outputs:

| Path | Content |
|---|---|
| `results/web/<label>.jsonl` | one record per request — the raw evidence |
| `results/web/<label>.json` | per-stage and overall summary |
| `results/web-bench.csv` | one row per run, machine-readable aggregate |
| `sites/<label>/` | the generated site — quality evidence, reviewed by hand |
| `logs/<label>.server.log` | server stderr, including acceptance-rate lines |
| `logs/<label>.agent.log` | full agent transcript |
| `logs/<label>.temps.log` | GPU telemetry |

### Why a proxy and not the server log

All three engines emit the same `timings` object in their OpenAI-compatible
responses — verified in `tools/server/server-task.cpp` for `pflash`/`buun` and
`examples/server/server-task.cpp` for `ik`. Their **log** formats differ. The
proxy is the one collector that works unchanged across all three, and it gives
per-request granularity that a summary log line cannot.

The proxy relays SSE chunks as they arrive rather than buffering, so it does not
distort the latency it is measuring.

---

## 4. Scoring quality

Throughput is only half the point. After each run, open the site and record in
[RESULTS.md](RESULTS.md):

| Check | Pass condition |
|---|---|
| Stage 1 | Site serves on its port; `<label>.service` is active |
| Stage 2 | Restyle applied; the JS interactive element actually works |
| Stage 3 | Game present, `public/game.js` loaded via `<script src>`, hook falls/reels on hold-release, fish are catchable, score increments |
| One-shot | Did the model complete each stage without asking a question? |
| Tool calls | Any malformed tool calls in `logs/<label>.agent.log`? |

Record the failure mode, not just pass/fail — "produced a game that never
increments the score" and "mangled every `write` call" are very different
verdicts about a config.

---

## 5. Port registry

Update this table when a run claims an index. Never reuse one.

| Index | Site port | Label | Engine / model / split | Status |
|---|---|---|---|---|
| _(none yet)_ | | | | |

---

## 6. Prerequisites

- `pi` on PATH (`/usr/bin/pi`, v0.78.0 at time of writing).
- **`PI_OFFLINE=1` is required** — the script sets it. Without it `pi` blocks on
  startup network operations in this environment and the stage hangs until its
  timeout. This was verified, not assumed.
- Node.js and `npm` available to the agent, plus sudo (the prompt supplies the
  password) for `systemctl`.
- The agent's `pi` config is written per-run into `sites/<label>/.pi-agent/`.
  The operator's `~/.pi/agent` is never touched.

### Tunables

| Env var | Default | When to change |
|---|---|---|
| `CTX` | `32768` | Drop to `16384` if the server OOMs on load. Agentic transcripts fill context fast, so don't go lower without noting it |
| `STAGE_TIMEOUT` | `3600` | Raise for slow configs. Exit code 124 from a stage means this was hit |
| `LOAD_TIMEOUT` | `900` | Raise under heavy disk contention |

A stage that hits `STAGE_TIMEOUT` is recorded as `FAILED(stageN)` and still
committed. **That is a legitimate result** — "could not finish the task in an
hour" is exactly the usability signal this phase exists to capture.
