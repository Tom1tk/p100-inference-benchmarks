#!/usr/bin/env python3
"""Regenerate Table A of the master results matrix from results/raw/*.csv.

Table A is the only generated part of RESULTS.md; B/C/D are hand-curated because
they carry judgement (invalid columns, failure modes, verdicts). Run this after
any new llama-bench sweep and paste the output over the existing Table A.

Notes on the parsing: raw files carry the llama-bench header (`build_commit,...`)
rather than all-results.csv's `label,...` form, some files contain more than one
header because a sweep was resumed, and h14-ubatch-sweep-hi has a crash backtrace
appended to it. Anything that doesn't parse into a row with a numeric avg_ts is
skipped rather than guessed at.
"""
import csv, glob, io, os
from collections import defaultdict

RAW = os.path.join(os.path.dirname(__file__), '..', 'results', 'raw')
ENGINES = {'e05ff58b7': 'PFlash*', '39d97a876': 'buun', '8337e4c': 'ik',
           '64f765f5': 'mainline', '57affa09': 'rebased'}
COLS = ['pp2048', 'pp4096', 'pp8192', 'pp16384', 'pp65536', 'pp100000', 'tg128']

def rows(path):
    header = None
    for line in open(path, errors='replace').read().splitlines():
        if line.startswith(('build_commit,', 'label,')):
            header = line
            continue
        if header is None or not line.strip():
            continue
        try:
            r = next(csv.DictReader(io.StringIO(header + '\n' + line)))
            float(r['avg_ts'])
        except (StopIteration, TypeError, ValueError, KeyError):
            continue
        yield r

def main():
    table, order = defaultdict(dict), []
    for path in sorted(glob.glob(os.path.join(RAW, '*.csv'))):
        for r in rows(path):
            quant = os.path.basename(r['model_filename']).replace('Qwen3.8-27B-UD-', '').replace('.gguf', '')
            # -dev CUDA0/CUDA1 means a single card; 'auto' or empty means both.
            gpus = '1' if (r.get('devices') or 'auto').strip() in ('CUDA0', 'CUDA1') else '2'
            label = os.path.basename(path)[:-4]
            engine = ENGINES.get(r['build_commit'], r['build_commit'])
            # H17 built the same commit twice; only the label separates them.
            if label.startswith('h17-patched'):
                engine += '+h17'
            key = (label, engine,
                   quant, r['split_mode'], gpus, r['n_batch'], r['n_ubatch'], r['type_k'])
            if key not in table:
                order.append(key)
            table[key]['tg128' if r['n_gen'] != '0' else 'pp' + r['n_prompt']] = float(r['avg_ts'])

    print('| Run | Engine | Quant | `-sm` | GPUs | `-b` | `-ub` | KV | '
          + ' | '.join(c.replace('pp', 'pp ') for c in COLS) + ' |')
    print('|---' * (8 + len(COLS)) + '|')
    for key in order:
        cells = [f"**{table[key][c]:.1f}**" if c in table[key] else '—' for c in COLS]
        print(f"| `{key[0]}` | " + ' | '.join(key[1:]) + ' | ' + ' | '.join(cells) + ' |')

if __name__ == '__main__':
    main()
