"""Summarise YOLOX-MTL stream runs from their eval.log files.

    python tools/summarize_mtl_final.py RUN_DIR [RUN_DIR ...]      # dirs that contain eval.log

Per run: mean AP50 / mAP / mIoU over domains (the AMROD protocol: mean of the per-domain scores) and, for
long-term streams (domain names repeating), the mean AP50 / mIoU per round.
"""
import re
import sys
from pathlib import Path

DOM = re.compile(r'\[DOMAIN (\d+)/(\d+)\] name=(\S+) n=(\d+) AP=([\w.]+) AP50=([\w.]+)')
SEG = re.compile(r'\[SEGDOMAIN (\d+)/(\d+)\] name=(\S+) n=(\d+) mIoU=([\w.]+)')


def parse(path):
    txt = Path(path).read_text(errors='ignore')
    det, seg = {}, {}
    for m in DOM.finditer(txt):
        det[int(m.group(1))] = (m.group(3), float(m.group(5)), float(m.group(6)))
    for m in SEG.finditer(txt):
        seg[int(m.group(1))] = float(m.group(5))
    n = max(det) if det else 0
    if not n or len(det) != n or len(seg) != n:
        return None
    names = [det[i][0] for i in range(1, n + 1)]
    ap = [det[i][1] for i in range(1, n + 1)]
    ap50 = [det[i][2] for i in range(1, n + 1)]
    miou = [seg[i] for i in range(1, n + 1)]
    return names, ap, ap50, miou


def mean(x):
    return 100.0 * sum(x) / len(x)


def main():
    print('| run | domains | AP50 | mAP | mIoU |\n|---|---|---|---|---|')
    extra = []
    for d in sys.argv[1:]:
        r = parse(Path(d) / 'eval.log')
        if r is None:
            print(f'| {Path(d).name} | unfinished / no metrics | | | |')
            continue
        names, ap, ap50, miou = r
        print(f'| {Path(d).name} | {len(names)} | {mean(ap50):.2f} | {mean(ap):.2f} | {mean(miou):.2f} |')
        period = next((k for k in range(1, len(names)) if names[k] == names[0] and names[:k] * (len(names) // k) == names), None)
        if period and len(names) > period:
            rounds = len(names) // period
            ap50_r = [mean(ap50[i * period:(i + 1) * period]) for i in range(rounds)]
            miou_r = [mean(miou[i * period:(i + 1) * period]) for i in range(rounds)]
            extra.append(f'{Path(d).name}\n  AP50 per round: ' + ', '.join(f'{v:.1f}' for v in ap50_r)
                         + '\n  mIoU per round: ' + ', '.join(f'{v:.1f}' for v in miou_r))
    if extra:
        print('\nLong-term rounds:\n' + '\n'.join(extra))


if __name__ == '__main__':
    main()
