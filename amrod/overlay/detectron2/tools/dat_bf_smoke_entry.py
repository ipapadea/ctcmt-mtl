"""Entry point for DAT-BF smoke tests on a truncated stream.

Registers ``<dataset>_first<N>`` (the first N records of ``<dataset>``, same metadata and
evaluator) and then runs the stock ``train_net.main``; nothing in the model, the evaluator or
the shared training code is changed. Dumps the online evaluator results to
``<OUTPUT_DIR>/online_results.json``. Smoke use only; full runs use tools/train_net.py.

  python tools/dat_bf_smoke_entry.py --config-file CFG --eval-only --num-gpus 1 \
      --truncate defocus_blur_semseg:100 OUTPUT_DIR ... SEED 0 DATASETS.TEST '("defocus_blur_semseg_first100",)'
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_net  # noqa: E402
from detectron2.data import DatasetCatalog, MetadataCatalog  # noqa: E402
from detectron2.engine import default_argument_parser, launch  # noqa: E402
from detectron2.utils import comm  # noqa: E402


def register_truncated(spec):
    name, _, n = spec.rpartition(":")
    n = int(n)
    new = f"{name}_first{n}"
    DatasetCatalog.register(new, lambda name=name, n=n: DatasetCatalog.get(name)[:n])
    meta = {k: v for k, v in MetadataCatalog.get(name).as_dict().items() if k != "name"}
    MetadataCatalog.get(new).set(**meta)
    return new


def main(args):
    res = train_net.main(args)
    if comm.is_main_process():
        cfg = train_net.setup(args)
        names = list(cfg.DATASETS.TEST)
        online = {names[0]: res} if len(names) == 1 else res
        with open(os.path.join(cfg.OUTPUT_DIR, "online_results.json"), "w") as f:
            json.dump(online, f, indent=2, default=float)
    return res


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--truncate", action="append", default=[], help="DATASET:N")
    args = parser.parse_args()
    for spec in args.truncate:
        print("registered", register_truncated(spec))
    launch(main, args.num_gpus, num_machines=args.num_machines, machine_rank=args.machine_rank,
           dist_url=args.dist_url, args=(args,))
