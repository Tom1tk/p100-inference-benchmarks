import re, sys, collections
sys.path.insert(0, '/root/llama.cpp-Pascal/gguf-py')
from gguf import GGUFReader
# Time budget per decode token (n=1) and per 2048-token prefill ubatch on ONE card, from measured op times.
model, sql = sys.argv[1], sys.argv[2]
t = {}
other = collections.defaultdict(float)
for line in open(sql):
    if not line.startswith('INSERT'): continue
    v = re.findall(r"'((?:[^']|'')*)'", line.split('VALUES', 1)[1])
    op, p, us = v[3], v[4], float(v[9])
    n = int(re.search(r'ne=\[[^,]+,(\d+)', p).group(1))
    mode = 'tg' if n == 1 or (op == 'GATED_DELTA_NET' and n == 129) else 'pp'
    if op == 'MUL_MAT':
        s = re.search(r'sources=(\w+)\[(\d+),(\d+)', p)
        t[(s.group(1).lower(), int(s.group(2)), int(s.group(3)), mode)] = us
    else:
        name = re.search(r'name=([^,]+)', p).group(1)
        t[(op, name, mode)] = us
r = GGUFReader(model)
bud = {m: collections.defaultdict(float) for m in ('tg', 'pp')}
miss = set()
n_layers = 0
for ten in r.tensors:
    m = re.match(r'blk\.(\d+)\.(\w+)\.weight$', ten.name)
    if not m: continue
    if int(m.group(1)) >= 64: continue  # MTP (nextn) block, not in the main forward pass
    n_layers = max(n_layers, int(m.group(1)) + 1)
    if len(ten.shape) != 2: continue
    k, o = int(ten.shape[0]), int(ten.shape[1])
    ty = ten.tensor_type.name.lower()
    fam = 'ffn' if m.group(2).startswith('ffn') else ('attn_full' if m.group(2) in ('attn_q','attn_k','attn_v','attn_output') else 'gdn_proj')
    for mode in ('tg', 'pp'):
        key = (ty, k, o, mode)
        if key in t: bud[mode]['GEMM/matvec ' + fam] += t[key]
        else: miss.add((m.group(2), ty, k, o))
# non-matmul ops: count per layer type
n_attn = sum(1 for x in r.tensors if x.name.endswith('attn_q.weight') and not x.name.startswith('blk.64.'))
n_gdn = sum(1 for x in r.tensors if x.name.endswith('ssm_out.weight'))
for mode in ('tg', 'pp'):
    for (key, us) in t.items():
        if len(key) != 3 or key[2] != mode: continue
        op = key[0]
        if op == 'GATED_DELTA_NET': bud[mode]['GDN scan'] += us * n_gdn
        elif op == 'FLASH_ATTN_EXT': bud[mode]['flash attn (4k ctx)'] += us * n_attn
        elif op == 'SSM_CONV': bud[mode]['ssm conv'] += us * n_gdn
print(f"{model.split('/')[-1]}: {n_layers} blocks, {n_attn} full-attn, {n_gdn} GDN; matvec/GEMM types from GGUF")
for mode, label in (('tg', 'decode, per token'), ('pp', 'prefill, per 2048-token ubatch')):
    tot = sum(bud[mode].values())
    print(f"  {label}: {tot/1000:.1f} ms measured-op total (excl. lm_head, norms, elementwise, launch gaps)")
    for k, us in sorted(bud[mode].items(), key=lambda x: -x[1]):
        print(f"    {k:28s} {us/1000:8.1f} ms  {100*us/tot:5.1f}%")
if miss: print('  unmatched:', sorted(miss)[:12])
