#!/usr/bin/env python3
"""Phase 0.2: per-phase time split from an nsys sqlite export of llama-bench (-p 2048 -n 64 -r 1, with warmup).
Windows are split on cuBLAS GEMM calls (see main)."""
import sqlite3, sys, re
from collections import defaultdict

CATS = [('gemm', r'gemm|cutlass|sgemm|hgemm|mul_mat_q|mmq'), ('matvec', r'mul_mat_vec|mmvq|mmvf'),
        ('fattn', r'flash_attn|fattn'), ('gdn', r'gated_delta|delta_net'), ('ssm_conv', r'ssm_conv'),
        ('dequant/convert', r'convert|dequantize|quantize|cpy'), ('allreduce add', r'add')]

def cat(name):
    for c, pat in CATS:
        if re.search(pat, name, re.I): return c
    return 'other'

def main(path):
    db = sqlite3.connect(path)
    k = db.execute("""select k.start,k.end,k.deviceId,s.value from CUPTI_ACTIVITY_KIND_KERNEL k
                      join StringIds s on k.demangledName=s.id order by k.start""").fetchall()
    try:
        m = db.execute("select start,end,deviceId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY order by start").fetchall()
    except sqlite3.OperationalError:
        m = []
    devs = sorted({r[2] for r in k})
    # llama-bench order: pp2048 warmup, pp2048, tg1 warmup, tg64; only pp issues cuBLAS GEMMs.
    # Measured pp = second half of dev0's GEMM calls (first GEMM start .. last GEMM end); tg = after that.
    g = [r for r in k if r[2] == devs[0] and cat(r[3]) == 'gemm']
    assert len(g) % 2 == 0, len(g)
    pp = (g[len(g) // 2][0], g[-1][1])
    tg = (pp[1], max(r[1] for r in k if r[2] == devs[0]))
    print(f"{path}: devices {devs}, {len(g)//2} GEMM calls per pp ubatch")
    windows = [pp, tg]
    for label, (s0, e0) in zip(['pp2048', 'tg64 (+tg1 warmup)'], windows):
        wall = (e0 - s0) / 1e6
        print(f"\n  {label}: wall {wall:.1f} ms" + (f" ({wall/65:.1f} ms/token)" if label.startswith('tg') else ''))
        for d in devs:
            kk = [r for r in k if r[2] == d and s0 <= r[0] <= e0]
            busy = defaultdict(float)
            for s, e, _, n in kk: busy[cat(n)] += (e - s) / 1e6
            mm = [r for r in m if r[2] == d and s0 <= r[0] <= e0]
            mt = sum((e - s) for s, e, *_ in mm) / 1e6; mb = sum(r[4] for r in mm) / 1e6
            tot = sum(busy.values())
            print(f"    dev{d}: {len(kk)} kernels, kernel busy {tot:.1f} ms ({100*tot/wall:.0f}% of wall), "
                  f"idle {wall-tot:.1f} ms | memcpy {len(mm)} ops {mt:.1f} ms {mb:.0f} MB")
            for c, v in sorted(busy.items(), key=lambda x: -x[1]):
                print(f"      {c:16s} {v:9.1f} ms {100*v/wall:5.1f}%")
    # top kernels overall in the tg window, dev0, to sanity-check categorisation
    s0, e0 = tg
    top = defaultdict(lambda: [0, 0.0])
    for s, e, d, n in k:
        if d == devs[0] and s0 <= s <= e0: top[n][0] += 1; top[n][1] += (e - s) / 1e6
    print("\n  top tg kernels dev0:")
    for n, (c, t) in sorted(top.items(), key=lambda x: -x[1][1])[:12]:
        print(f"    {t:8.1f} ms {c:6d}x [{cat(n)}] {n[:110]}")

if __name__ == '__main__':
    main(sys.argv[1])
