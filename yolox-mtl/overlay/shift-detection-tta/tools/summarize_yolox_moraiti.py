#!/usr/bin/env python3
"""Source-only vs Moraiti MT+CL+SR (YOLOX-X, Cityscapes checkpoint of Moraiti et al.).

Reads the evaluation logs written by docker/run_leviathan_eval.sh (lines
``[DOMAIN i/n] name=... AP=... AP50=...`` of CocoMetricDomains) and writes per_domain.csv,
per_protocol.csv and summary.md into --out.

Source-only is evaluated once on ACDC4 and CSC12.  The long-term streams (ACDC-LT, CSC-LT) repeat
the same domains, and without adaptation the result of a domain does not depend on its position, so
the Source-only long-term rows are those domain results repeated (not separate runs).

    python3 tools/summarize_yolox_moraiti.py --src DIR --mor DIR --out DIR
"""
import argparse
import csv
import os
import re
import statistics as st

PAT = re.compile(r'\[DOMAIN (\d+)/(\d+)\] name=(\S+) n=(\d+) AP=([\d.\-]+) AP50=([\d.\-]+) AP75=([\d.\-]+)')
CSC_CYCLE = ['fog', 'motion_blur', 'snow', 'brightness', 'defocus_blur']
PROTOCOLS = [('ACDC4', 'acdc4', 4), ('CSC12', 'csc12', 12), ('ACDC-LT', 'acdcLT', 40), ('CSC-LT', 'cscLT', 50)]


def parse(path):
    rows = []
    for line in open(path, errors='ignore'):
        m = PAT.search(line)
        if m:
            rows.append((m.group(3), int(m.group(4)), float(m.group(5)) * 100, float(m.group(6)) * 100))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='dir with the source-only run dirs')
    ap.add_argument('--mor', required=True, help='dir with the Moraiti run dirs')
    ap.add_argument('--out', default='results/yolox_moraiti')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    src = {'acdc4': parse(os.path.join(args.src, 'src_acdc4/eval.log')),
           'csc12': parse(os.path.join(args.src, 'src_csc12/eval.log'))}
    src['acdcLT'] = src['acdc4'] * 10
    by_name = {r[0]: r for r in src['csc12']}
    src['cscLT'] = [by_name[n] for n in CSC_CYCLE] * 10
    mor = {key: parse(os.path.join(args.mor, f'moraiti_{key}_s1/eval.log')) for _, key, _ in PROTOCOLS}

    per_domain, per_protocol = [], []
    for label, key, n in PROTOCOLS:
        s, m = src[key], mor[key]
        assert len(s) == n and len(m) == n, (label, len(s), len(m))
        assert [r[0] for r in s] == [r[0] for r in m], f'{label}: domain order differs'
        for i, (a, b) in enumerate(zip(s, m)):
            per_domain.append([label, i, a[0], a[1], f'{a[2]:.3f}', f'{a[3]:.3f}', f'{b[2]:.3f}', f'{b[3]:.3f}'])
        per_protocol.append([label, n, st.mean(r[2] for r in s), st.mean(r[3] for r in s),
                             st.mean(r[2] for r in m), st.mean(r[3] for r in m)])

    with open(os.path.join(args.out, 'per_domain.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['protocol', 'index', 'domain', 'n_images', 'source_mAP', 'source_AP50', 'moraiti_mAP', 'moraiti_AP50'])
        w.writerows(per_domain)
    with open(os.path.join(args.out, 'per_protocol.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['protocol', 'n_domains', 'source_mAP', 'source_AP50', 'moraiti_mAP', 'moraiti_AP50'])
        w.writerows([[p[0], p[1]] + [f'{x:.3f}' for x in p[2:]] for p in per_protocol])

    def rounds(rows, k):
        return [st.mean(r[3] for r in rows[i:i + k]) for i in range(0, len(rows), k)]

    L = ['# YOLOX-X: Source-only vs MT + CL + SR (Moraiti et al.)', '',
         'Checkpoint: `yolox_x_8xb4-24e_cityscapes.pth` of the authors (md5 `f79e0f1f6cbc56513756ddd1d1dfaeb0`). '
         'Seed 0 (config `seed=1234`), one run per protocol. Official adapter `MeanTeacherYOLOXAdapterContrastive` with the '
         'hyper-parameters of the committed Cityscapes config (optim_steps=1, EMA 0.99965, consistency 0.005, contrastive 0.003, '
         'restoration p=0.01, 2 views, SGD lr 0.00025). Mean over the domains of the stream.', '',
         '| Protocol | Source AP50 | Moraiti AP50 | Δ AP50 | Source mAP | Moraiti mAP | Δ mAP |', '|---|---|---|---|---|---|---|']
    for p in per_protocol:
        L.append(f'| {p[0]} | {p[3]:.2f} | {p[5]:.2f} | {p[5] - p[3]:+.2f} | {p[2]:.2f} | {p[4]:.2f} | {p[4] - p[2]:+.2f} |')
    L += ['', '## Long-term: mean AP50 per round (Moraiti)', '']
    L.append('- ACDC-LT (4 domains/round): ' + ', '.join(f'{x:.1f}' for x in rounds(mor['acdcLT'], 4)))
    L.append('- CSC-LT (5 domains/round): ' + ', '.join(f'{x:.1f}' for x in rounds(mor['cscLT'], 5)))
    L += ['', '## CSC12 per domain (AP50)', '', '| Domain | Source | Moraiti | Δ |', '|---|---|---|---|']
    for a, b in zip(src['csc12'], mor['csc12']):
        L.append(f'| {a[0]} | {a[3]:.1f} | {b[3]:.1f} | {b[3] - a[3]:+.1f} |')
    L += ['', '## ACDC4 per domain (AP50)', '', '| Domain | Source | Moraiti | Δ |', '|---|---|---|---|']
    for a, b in zip(src['acdc4'], mor['acdc4']):
        L.append(f'| {a[0]} | {a[3]:.1f} | {b[3]:.1f} | {b[3] - a[3]:+.1f} |')
    L += ['', '## Caveats', '',
          '- Cross-architecture: YOLOX-X has its own, much higher, Source than the Panoptic-FPN R50 of the CTCMT tables; '
          'only Source vs MT+CL+SR on the same YOLOX is a like-for-like comparison.',
          '- Detection only, one seed, no error bars.',
          '- optim_steps=1 as in the committed config; the paper text says 5.',
          '- Class order: mmdet takes labels from the *order of `categories` in the json*, so the jsons were re-ordered to the '
          'checkpoint order (`tools/data/reorder_coco_categories.py`). Without that five classes score 0.',
          '- Source on clean Cityscapes val is 42.2 mAP here vs 40.3 in the paper (Table 5); cause not determined.',
          '- The long-term Source rows are the short-term domain results repeated, not separate runs.']
    open(os.path.join(args.out, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
