#!/usr/bin/env python3
import os

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser, launch
from detectron2.evaluation import verify_results

from train_net import Trainer as BaseTrainer, build_evaluator, setup


class DumpTrainer(BaseTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        # Important difference from normal train_net.py:
        # preserve predictions from EVERY dataset in the continual stream.
        if output_folder is None:
            output_folder = os.path.join(
                cfg.OUTPUT_DIR,
                "inference_by_dataset",
                dataset_name,
            )
        return build_evaluator(cfg, dataset_name, output_folder)


def main(args):
    cfg = setup(args)

    if not args.eval_only:
        raise RuntimeError("paper26_dump_predictions.py is eval-only")

    model = DumpTrainer.build_model(cfg)

    # Same self-loading convention as the canonical train_net.py.
    self_loading = {
        "AMROD",
        "CoTTA_SemSeg",
        "CTCMT_MTL",
        "CTCMT_Seg",
        "TENT_SemSeg",
    }

    if cfg.MODEL.META_ARCHITECTURE not in self_loading:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS,
            resume=args.resume,
        )

    res = DumpTrainer.test(cfg, model)

    if cfg.TEST.AUG.ENABLED:
        res.update(DumpTrainer.test_with_TTA(cfg, model))

    if comm.is_main_process():
        verify_results(cfg, res)

    return res


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
