#!/usr/bin/env python3
"""Compare greedy outputs of run-spec-placement arms (rep 1 of each results/h11/<label>.jsonl).

Compares reasoning_content + content. content alone is usually empty: within 400 tokens
the model is still thinking, so a content-only diff is vacuous (H27 correction, 2026-10-14).
Usage: compare-outputs.py LABEL_A LABEL_B [LABEL_A LABEL_B ...]
"""
import json, sys

def text(label):
    for line in open(f"results/h11/{label}.jsonl"):
        if line.strip():
            m = json.loads(line)["choices"][0]["message"]
            return (m.get("reasoning_content") or "") + (m.get("content") or "")
    return None

args = sys.argv[1:]
assert args and len(args) % 2 == 0, __doc__
for a, b in zip(args[::2], args[1::2]):
    x, y = text(a), text(b)
    if x is None or y is None:
        print(f"{a} {b} EMPTY")
        continue
    d = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None if len(x) == len(y) else min(len(x), len(y)))
    print(f"{a} {b} {'identical' if d is None else f'differ@{d}'} ({len(x)},{len(y)} chars)")
