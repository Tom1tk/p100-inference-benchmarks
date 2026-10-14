#!/usr/bin/env python3
"""H19 client: grow one chat session turn by turn, record what the server re-prefills.

Turns 1-4 grow the main session. Then an unrelated request takes the only slot,
turn 5 comes back to the main session (does the idle-slot cache bring it back?),
and a final control re-sends turn 5 with cache_prompt=false (cold prefill).
"""
import csv, json, sys, time, urllib.request

port, prompt_file, out_csv, label = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
URL = f"http://127.0.0.1:{port}/v1/chat/completions"
SYSTEM = open(prompt_file).read()
ASKS = [
    "Summarise the document above in three sentences.",
    "Which section is the longest, and why might that be?",
    "List five specific numbers that appear in it.",
    "Suggest one improvement to the document's structure.",
    "Now give a one-line title for it.",
]

def ask(messages, cache=True):
    body = {"messages": messages, "max_tokens": 128, "temperature": 0, "cache_prompt": cache,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    t = r["timings"]
    return r["choices"][0]["message"]["content"], t, time.time() - t0

rows = []
def record(step, t, wall):
    total = t.get("cache_n", 0) + t["prompt_n"]
    row = dict(label=label, step=step, prompt_total=total, cache_n=t.get("cache_n", 0),
               prompt_n=t["prompt_n"], prompt_ms=round(t["prompt_ms"], 1),
               decode_tps=round(t.get("predicted_per_second", 0), 2), wall_s=round(wall, 1))
    rows.append(row); print(row, flush=True)

msgs = [{"role": "system", "content": SYSTEM}]
for i, q in enumerate(ASKS[:4], 1):
    msgs.append({"role": "user", "content": q})
    a, t, w = ask(msgs); record(f"turn{i}", t, w)
    msgs.append({"role": "assistant", "content": a})

_, t, w = ask([{"role": "user", "content": "Write a haiku about GPUs. " * 40}]); record("interleave", t, w)

msgs.append({"role": "user", "content": ASKS[4]})
_, t, w = ask(msgs); record("turn5-after-interleave", t, w)
_, t, w = ask(msgs, cache=False); record("turn5-cold-control", t, w)

with open(out_csv, "a", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=rows[0].keys())
    if f.tell() == 0: wr.writeheader()
    wr.writerows(rows)
