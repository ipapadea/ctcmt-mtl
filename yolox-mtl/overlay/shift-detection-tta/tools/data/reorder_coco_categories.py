#!/usr/bin/env python3
"""Write copies of COCO detection jsons whose `categories` list follows a given class order.

mmdet's CocoDataset maps category ids to labels with ``coco.get_cat_ids(cat_names=classes)``,
which returns the ids in the order of the json's `categories` list, NOT in the order of
`metainfo.classes`.  A detector trained with another class order therefore sees permuted
ground-truth labels.  The AMROD / ACDC jsons list the classes as
person, rider, car, truck, bus, train, motorcycle, bicycle, whereas the YOLOX checkpoints of
shift-detection-tta use person, rider, car, train, motorcycle, bicycle, truck, bus.

Only the order of `categories` changes (ids, names, images and annotations are untouched);
the originals are left alone.

    python3 tools/data/reorder_coco_categories.py IN.json [IN2.json ...] [--suffix _yoloxorder]
"""
import argparse
import json
import os

YOLOX_ORDER = ['person', 'rider', 'car', 'train', 'motorcycle', 'bicycle', 'truck', 'bus']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--order', nargs='+', default=YOLOX_ORDER)
    ap.add_argument('--suffix', default='_yoloxorder')
    args = ap.parse_args()

    for path in args.inputs:
        with open(path) as f:
            data = json.load(f)
        by_name = {c['name']: c for c in data['categories']}
        assert sorted(by_name) == sorted(args.order), (path, sorted(by_name))
        data['categories'] = [by_name[n] for n in args.order]
        root, ext = os.path.splitext(path)
        out = root + args.suffix + ext
        with open(out, 'w') as f:
            json.dump(data, f)
        print(f'{path} -> {out}  ({len(data["images"])} images, {len(data["annotations"])} annotations)')


if __name__ == '__main__':
    main()
