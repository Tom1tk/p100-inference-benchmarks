#!/usr/bin/env python3
"""Phase 8 tier 2 fixture: one ~98k-token haystack, many questions (QUALITY-PLAN §2.4).

Haystack = prompts/h26-100k-prompt.txt minus its trailing question, with 5 needles planted
as code comments at ~10/25/50/75/90% depth. Questions:
  layer 1 - recall each needle's number (not guessable);
  layer 2 - name the function that contains a quoted line (needs locating + reading upward).
Every answer is checked by substring match. Deterministic: same output on every run.
Writes prompts/phase8-fixture.json.
"""
import json, random, re

SRC = "prompts/h26-100k-prompt.txt"
OUT = "prompts/phase8-fixture.json"
NEEDLE_DEPTHS = [10, 25, 50, 75, 90]
NEEDLE_KEYS = ["heron", "quartz", "lantern", "orchid", "anvil"]
FUNC_DEPTHS = [5, 15, 22, 30, 38, 45, 52, 58, 65, 72, 80, 86, 92, 96]

rng = random.Random(8)
text = open(SRC).read()
text = text[: text.index("===== END OF CORPUS =====") + len("===== END OF CORPUS =====")] + "\n"
lines = text.split("\n")
n = len(lines)

fdef = re.compile(r"^(?:static\s+)?[\w:<>\*&\s,]+?\b([A-Za-z_][\w:]*)\s*\([^;]*\)\s*(const\s*)?\{\s*$")
funcs = []  # (start, end, name) of top-level functions; end = first line that is exactly "}"
for i, l in enumerate(lines):
    m = fdef.match(l)
    if m and not l[0].isspace():
        end = next((j for j in range(i + 1, n) if lines[j] == "}"), None)
        if end:
            funcs.append((i, end, m.group(1)))

counts = {}
for l in lines:
    counts[l.strip()] = counts.get(l.strip(), 0) + 1

questions = []
used = set()
for d in FUNC_DEPTHS:
    target = n * d // 100
    for s, e, name in sorted(funcs, key=lambda f: abs(f[0] - target)):
        short = name.split("::")[-1]
        if name in used or e - s < 12:
            continue
        cands = [lines[k].strip() for k in range(s + 3, e - 1)
                 if counts[lines[k].strip()] == 1 and 30 <= len(lines[k].strip()) <= 110
                 and short not in lines[k] and "//" not in lines[k]]
        if not cands:
            continue
        used.add(name)
        q_line = rng.choice(cands)
        questions.append({"layer": 2, "depth": round(s / n * 100), "answer": short,
                          "q": f"In the corpus, which function's body contains this exact line?\n\n"
                               f"    {q_line}\n\nReply with only the function name, nothing else."})
        break

# Plant needles last, between top-level definitions, from the end so indices stay valid.
needles = []
for d, key in zip(NEEDLE_DEPTHS, NEEDLE_KEYS):
    target = n * d // 100
    at = min((e + 1 for _, e, _ in funcs), key=lambda k: abs(k - target))
    needles.append((at, d, key, str(rng.randint(100000, 999999))))
for at, d, key, code in sorted(needles, reverse=True):
    lines.insert(at, f"// Maintainer note: the calibration constant for the {key} subsystem is {code}.")
for at, d, key, code in needles:
    questions.append({"layer": 1, "depth": d, "answer": code,
                      "q": f"According to the maintainer notes in the corpus, what is the calibration "
                           f"constant for the {key} subsystem? Reply with only the number, nothing else."})

questions.sort(key=lambda q: (q["layer"], q["depth"]))
json.dump({"system": "\n".join(lines), "questions": questions}, open(OUT, "w"), indent=1)
print(f"wrote {OUT}: {len(questions)} questions")
for q in questions:
    print(q["layer"], q["depth"], q["answer"])
