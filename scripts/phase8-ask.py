#!/usr/bin/env python3
"""Ask every question in prompts/phase8-fixture.json against a running llama-server.
Usage: phase8-ask.py <label> <port>. Appends rows to results/phase8-tier2.csv and the raw
replies to results/phase8/<label>.jsonl. Exits 3 if question 2 re-prefills the haystack
(prompt cache not reusing the prefix: the sweep would cost 8 min per question)."""
import csv, json, os, sys, time, urllib.request

label, port = sys.argv[1], sys.argv[2]
fx = json.load(open("prompts/phase8-fixture.json"))
os.makedirs("results/phase8", exist_ok=True)
new = not os.path.exists("results/phase8-tier2.csv")
out = csv.writer(open("results/phase8-tier2.csv", "a", newline=""))
if new:
    out.writerow(["label", "layer", "depth", "answer", "hit", "reply", "prompt_n", "prompt_ms"])
raw = open(f"results/phase8/{label}.jsonl", "w")

for i, q in enumerate(fx["questions"]):
    body = {"messages": [{"role": "system", "content": fx["system"]}, {"role": "user", "content": q["q"]}],
            "temperature": 0, "top_k": 1, "seed": 42, "max_tokens": 48, "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    reply = r["choices"][0]["message"].get("content") or ""
    t = r.get("timings", {})
    hit = q["answer"] in reply
    out.writerow([label, q["layer"], q["depth"], q["answer"], int(hit), reply.strip()[:80],
                  t.get("prompt_n"), round(t.get("prompt_ms", 0))])
    raw.write(json.dumps(r) + "\n"); raw.flush()
    print(f"{label} L{q['layer']} d{q['depth']:>2} {'HIT ' if hit else 'MISS'} {reply.strip()[:40]!r} "
          f"prompt_n={t.get('prompt_n')} {time.time() - t0:.0f}s", flush=True)
    if i == 1 and (t.get("prompt_n") or 0) > 5000:
        print("ABORT: question 2 re-prefilled the haystack; prefix cache not reused", flush=True)
        sys.exit(3)
