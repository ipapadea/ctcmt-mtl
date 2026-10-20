#!/usr/bin/env python3

import argparse
import csv
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn

from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import inference_on_dataset
from detectron2.modeling.postprocessing import (
    detector_postprocess,
    sem_seg_postprocess,
)
from detectron2.utils.env import seed_all_rng

from train_net import Trainer


DEFAULT_CYCLE = [
    "fog_mtl",
    "motion_blur_mtl",
    "snow_mtl",
    "brightness_mtl",
    "defocus_blur_mtl",
]


def parse_csv_list(s):
    if s is None or not s.strip():
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


class FrozenTeacherProbe(nn.Module):
    """
    Read-only view of the current EMA teacher.

    Important:
      - no student forward
      - no pseudo labels
      - no loss
      - no optimizer
      - no EMA
      - no stochastic restore
      - does not increment CTCMT_MTL.iter
    """

    def __init__(self, teacher):
        super().__init__()

        # Do not register the same teacher as a child module of this wrapper.
        # We only keep a Python reference to it.
        object.__setattr__(self, "_teacher_ref", teacher)

        teacher.eval()
        self.eval()

    @torch.no_grad()
    def forward(self, batched_inputs):
        teacher = object.__getattribute__(self, "_teacher_ref")
        teacher.eval()

        # Needed only for the resized image sizes used by postprocessing.
        images = teacher.preprocess_image(batched_inputs)

        out = teacher.inference(
            batched_inputs,
            do_postprocess=False,
        )

        if isinstance(out, tuple) and len(out) == 2:
            t_det, t_sem = out
        else:
            t_det, t_sem = out, None

        processed = []

        for i, (inp, image_size) in enumerate(
            zip(batched_inputs, images.image_sizes)
        ):
            H = inp.get("height", image_size[0])
            W = inp.get("width", image_size[1])

            item = {}

            if t_det is not None:
                item["instances"] = detector_postprocess(
                    t_det[i], H, W
                )

            if t_sem is not None:
                item["sem_seg"] = sem_seg_postprocess(
                    t_sem[i],
                    image_size,
                    H,
                    W,
                )

            processed.append(item)

        return processed


def extract_metrics(result):
    ap50 = float("nan")
    miou = float("nan")

    if isinstance(result, dict):
        bbox = result.get("bbox")
        if isinstance(bbox, dict):
            ap50 = float(bbox.get("AP50", float("nan")))

        sem = result.get("sem_seg")
        if isinstance(sem, dict):
            miou = float(sem.get("mIoU", float("nan")))

    return ap50, miou


def run_eval(cfg, model, dataset_name, output_folder):
    loader = build_detection_test_loader(cfg, dataset_name)

    evaluator = Trainer.build_evaluator(
        cfg,
        dataset_name,
        output_folder=output_folder,
    )

    return inference_on_dataset(
        model,
        loader,
        evaluator,
    )


def append_csv(path, row):
    exists = path.exists()

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "kind",
                "state_idx",
                "round",
                "position",
                "adapted_dataset",
                "probe_dataset",
                "iter_before",
                "iter_after",
                "AP50",
                "mIoU",
            ],
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)

def save_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--config-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--cycle",
        default=",".join(DEFAULT_CYCLE),
        help="Comma-separated adaptation sequence.",
    )

    p.add_argument("--rounds", type=int, default=1)

    p.add_argument(
        "--probe-after-each",
        default="",
        help="Comma-separated frozen probe datasets after every adaptation domain.",
    )

    p.add_argument(
        "--round-end-all",
        action="store_true",
        help="At the end of each round frozen-probe every dataset in the cycle.",
    )

    p.add_argument(
        "--final-probes",
        default="",
        help="Comma-separated frozen probes after the complete stream.",
    )

    args = p.parse_args()

    cycle = parse_csv_list(args.cycle)
    dense_probes = parse_csv_list(args.probe_after_each)
    final_probes = parse_csv_list(args.final_probes)

    if not cycle:
        raise ValueError("Empty adaptation cycle.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "retention.csv"

    seed_all_rng(args.seed)

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.OUTPUT_DIR = str(out)
    cfg.SEED = args.seed
    cfg.freeze()

    print("=" * 100)
    print("RETENTION PROBE")
    print(f"method            : {args.method}")
    print(f"config            : {args.config_file}")
    print(f"seed              : {args.seed}")
    print(f"rounds            : {args.rounds}")
    print(f"cycle             : {cycle}")
    print(f"probe-after-each  : {dense_probes}")
    print(f"round-end-all     : {args.round_end_all}")
    print(f"final-probes      : {final_probes}")
    print("=" * 100)

    # CTCMT_MTL from_config loads student / teacher / anchor weights itself.
    model = Trainer.build_model(cfg)

    if not hasattr(model, "teacher"):
        raise RuntimeError(
            "Retention probe expects a CTTA model exposing model.teacher"
        )

    probe_model = FrozenTeacherProbe(model.teacher)

    eval_counter = 0
    state_idx = 0

    def evaluate(
        eval_model,
        dataset,
        kind,
        round_idx,
        position,
        adapted_dataset,
        probe_dataset,
    ):
        nonlocal eval_counter

        eval_counter += 1

        tag = (
            f"{eval_counter:03d}_"
            f"{kind}_"
            f"r{round_idx}_"
            f"p{position}_"
            f"{dataset}"
        )

        folder = out / tag

        iter_before = int(getattr(model, "iter", -1))

        print()
        print("-" * 100)
        print(
            f"[RETENTION] kind={kind} "
            f"state={state_idx} "
            f"round={round_idx} "
            f"position={position} "
            f"dataset={dataset} "
            f"iter_before={iter_before}"
        )

        # result = run_eval(
        #     cfg,
        #     eval_model,
        #     dataset,
        #     str(folder),
        # )
        if kind.startswith("probe"):
            rng_state = save_rng_state()
            try:
                result = run_eval(
                    cfg,
                    eval_model,
                    dataset,
                    str(folder),
                )
            finally:
                restore_rng_state(rng_state)
        else:
            result = run_eval(
                cfg,
                eval_model,
                dataset,
                str(folder),
            )

        iter_after = int(getattr(model, "iter", -1))

        # A frozen probe must not execute CTCMT_MTL.forward().
        if kind.startswith("probe") and iter_after != iter_before:
            raise RuntimeError(
                f"Probe modified adaptation iter: "
                f"{iter_before} -> {iter_after}"
            )

        ap50, miou = extract_metrics(result)

        row = {
            "method": args.method,
            "kind": kind,
            "state_idx": state_idx,
            "round": round_idx,
            "position": position,
            "adapted_dataset": adapted_dataset,
            "probe_dataset": probe_dataset,
            "iter_before": iter_before,
            "iter_after": iter_after,
            "AP50": ap50,
            "mIoU": miou,
        }

        append_csv(csv_path, row)

        with (folder / "result.json").open("w") as f:
            json.dump(result, f, indent=2)

        print(
            f"[RETENTION-RESULT] "
            f"kind={kind} "
            f"state={state_idx} "
            f"dataset={dataset} "
            f"AP50={ap50:.6f} "
            f"mIoU={miou:.6f} "
            f"iter={iter_before}->{iter_after}"
        )

    for round_idx in range(1, args.rounds + 1):
        for position, adapt_dataset in enumerate(cycle, start=1):
            state_idx += 1

            # ----------------------------------------------------------
            # ADAPTATION evaluation: normal CTCMT model.
            # ----------------------------------------------------------
            evaluate(
                model,
                adapt_dataset,
                "adapt",
                round_idx,
                position,
                adapt_dataset,
                "",
            )

            # ----------------------------------------------------------
            # Frozen probes immediately after this adaptation state.
            # ----------------------------------------------------------
            for probe_dataset in dense_probes:
                evaluate(
                    probe_model,
                    probe_dataset,
                    "probe_after_each",
                    round_idx,
                    position,
                    adapt_dataset,
                    probe_dataset,
                )

            # ----------------------------------------------------------
            # Full retention snapshot at round boundary.
            # Avoid re-running a dense probe already done at this state.
            # ----------------------------------------------------------
            if args.round_end_all and position == len(cycle):
                for probe_dataset in cycle:
                    if probe_dataset in dense_probes:
                        continue

                    evaluate(
                        probe_model,
                        probe_dataset,
                        "probe_round_end",
                        round_idx,
                        position,
                        adapt_dataset,
                        probe_dataset,
                    )

    for probe_dataset in final_probes:
        evaluate(
            probe_model,
            probe_dataset,
            "probe_final",
            args.rounds,
            len(cycle),
            cycle[-1],
            probe_dataset,
        )

    print()
    print("=" * 100)
    print(f"DONE: {args.method}")
    print(f"CSV : {csv_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
