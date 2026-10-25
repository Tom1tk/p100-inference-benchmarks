# Gemma 4 QAT (12B and E4B)

A second experiment beside the Qwen3.8-27B work: dial in decode and prefill speed for two Gemma 4 QAT models, with
the same target as the main objective (fastest decode and prefill at 100k context, one llama-server command) and
the head start of everything the fork already does for Pascal. Opened 2026-10-25 (user call).

Runs are numbered **G1, G2, …** so they never collide with the H-series in [HYPOTHESES.md](HYPOTHESES.md). The
standing rules all apply: rule zero, 200 W cap, 80 °C abort, one lever per run, and commit and push after every
run.

**Single card is in scope here** (user call, 2026-10-25), unlike the Qwen track. Both models fit on one 16 GB
card, and the Qwen work showed that the two-card cost is host-side sync, which a small model can't hide. Tests
run on **card 1**, through `CVD=1` in `scripts/run-spec-placement.sh`.

## Models

All files live in `/root/gemma4/`.

| Role | File | Size | Source |
|---|---|---|---|
| 12B target | `gemma-4-12b-it-qat-q4_0.gguf` | 6.98 GB | [google/gemma-4-12B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf) |
| E4B target | `gemma-4-E4B_q4_0-it.gguf` | 5.15 GB | [google/gemma-4-E4B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf) |
| 12B drafter (Unsloth) | `mtp-gemma-4-12B-it.gguf` | 254 MB | [unsloth/gemma-4-12B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF) |
| E4B drafter (Unsloth) | `mtp-gemma-4-E4B-it.gguf` | 60 MB | [unsloth/gemma-4-E4B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF) |
| 12B drafter (QAT) | `mtp-qat-gemma-4-12B-it-Q4_0.gguf` | 293 MiB | converted from [google/gemma-4-12B-it-qat-q4_0-unquantized-assistant](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized-assistant) |
| E4B drafter (QAT) | `mtp-qat-gemma-4-E4B-it-Q4_0.gguf` | 58 MiB | converted from [google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant) |

**Why Google's Q4_0 and not Unsloth's UD-Q4_K_XL:**
- QAT trains the model for exactly Q4_0, so Google's file is the format the training targeted.
- Q4_0 is the simplest type to decode.
- The H65 "straight Q4_0 closed" verdict was for non-QAT Qwen and does not carry over.
- The Unsloth target files were deleted.

**Drafters.** Neither Google GGUF has a built-in drafter. The 12B header was checked: `blk.0`–`blk.47` only, no MTP
tensors. Google's model card says a QAT target needs a QAT assistant of the same precision. The Unsloth drafter's
metadata names it plain "12B Assistant", which suggests it came from the non-QAT checkpoint (inferred, not
confirmed). G2 settles which drafter is better.

**Converting the QAT drafters.** This is CPU only. The fork's converter already registers
`Gemma4UnifiedAssistantForCausalLM`. Torch comes from `/opt/hindsight-venv`.

```
hf download google/gemma-4-12B-it-qat-q4_0-unquantized-assistant --local-dir /root/gemma4/qat-assistant-12B
/opt/hindsight-venv/bin/python /root/llama.cpp-Pascal/convert_hf_to_gguf.py /root/gemma4/qat-assistant-12B \
    --outtype f16 --outfile /root/gemma4/mtp-qat-gemma-4-12B-it-F16.gguf
llama-quantize /root/gemma4/mtp-qat-gemma-4-12B-it-F16.gguf /root/gemma4/mtp-qat-gemma-4-12B-it-Q4_0.gguf Q4_0
```

The same commands with `E4B` give the E4B drafter.

## Engine

The engine is our fork, [Tom1tk/llama.cpp-Pascal](https://github.com/Tom1tk/llama.cpp-Pascal) at `a563b40`, built in
`build-cuda-p100`. It needs no changes for Gemma 4:
- The `gemma4` and `gemma4-assistant` architectures and the shared-KV MTP path are already upstream.
- The 512-wide global attention heads use the FA tile kernel on Pascal, which has `dkq512` instances.
- The Pascal FA GEMM prefill path is head-size generic.
- Qwen-only levers (gated delta-net, H84 KV-only catch-up) don't apply. Gemma's drafter shares the target KV and
  skips catch-up anyway.

## Runs

The prompt is `prompts/h11-depth-prompt.txt`, about 11k tokens. Requests use temp 0, top_k 1, 400 tokens. Settings
are 16k ctx, f16 KV, `-fa 1`, `-b/-ub 2048`, `-sm none`, card 1.

| Run | Question | Pre-registered expectation | Result |
|---|---|---|---|
| G1 | Baselines: Google Q4_0 + Unsloth drafter, 12B then E4B, one run each (`scripts/run-g1-gemma4.sh`) | Loads and serves on one card; decode well above the Qwen 2-card 57.9 t/s | **12B: 44.6 t/s decode, 396 t/s prefill, accept 65.7%, 10.2 GB. E4B: 84.3 / 1054 t/s, 59.3%, 4.2 GB.** Both serve on one card. The 12B is below the Qwen 2-card decode, and its prefill is the obvious lever |
| G2 | One lever vs G1: the drafter file, Unsloth → our QAT conversion (`scripts/run-g2-qat-drafter.sh`) | QAT drafter acceptance ≥ Unsloth's on the same output; the faster drafter goes in the serve command | **No win; keep Unsloth.** 12B: 43.5 / 400 t/s, 64.9% (vs 44.6 / 396, 65.7%). E4B: 80.9 / 1052 t/s, 59.3% (same 255/430 accepted, about 4% slower decode). Single runs; the gaps are noise or slightly against QAT |

**Next, once a drafter is chosen:**
- 100k ctx on one card (VRAM, prefill, decode).
- One two-card A/B for 12B prefill only.
- A KLD check of Q4_0 against a BF16 reference.
- Then Gemma's recommended sampler settings (temp 1.0, top_k 64, top_p 0.95), which change the sampler path.
