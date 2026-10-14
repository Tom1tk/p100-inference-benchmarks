import re, sys
# Summarise test-backend-ops SQL perf output: per op, time, TFLOPS, GB/s.
rows = []
for line in open(sys.argv[1]):
    if not line.startswith('INSERT'):
        continue
    v = re.findall(r"'((?:[^']|'')*)'", line.split('VALUES', 1)[1])
    op, params, t, fl, bw = v[3], v[4], float(v[9]), float(v[10]), float(v[11])
    rows.append((op, params, t, fl, bw))
want = sys.argv[2].split(',') if len(sys.argv) > 2 else None
for op, params, t, fl, bw in rows:
    if want and op not in want:
        continue
    name = re.search(r'name=([^,]+)', params).group(1)
    ty = re.search(r'type=([^,]+)', params).group(1)
    ne = re.search(r'ne=\[([^\]]+)\]', params).group(1)
    src = re.search(r'sources=(.*)$', params).group(1)[:70]
    print(f"{op:16s} {name:22s} ne=[{ne:22s}] {t:10.1f} us {fl/1e12:7.2f} TFLOPS {bw:7.1f} GB/s  {src}")
