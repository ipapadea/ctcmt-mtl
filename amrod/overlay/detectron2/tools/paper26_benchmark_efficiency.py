#!/usr/bin/env python3

import argparse
import gc
import json
import math
import os
import platform
import random
import statistics
import time

import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.utils.events import EventStorage

from train_net import Trainer


def percentile(xs, q):
    if not xs:
        return float("nan")
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


def count_params(module):
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters())


def count_trainable(module):
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def module_bytes(module):
    if module is None:
        return 0

    total = 0

    for p in module.parameters():
        total += p.numel() * p.element_size()

    for b in module.buffers():
        total += b.numel() * b.element_size()

    return total


def mib(x):
    return float(x) / (1024.0 ** 2)


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--config-file", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--dataset", default="defocus_blur_mtl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeat", type=int, default=1)

    # 50 + 400 = 450 / 500 Cityscapes-val images.
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--measure", type=int, default=400)

    p.add_argument("--output-json", required=True)
    p.add_argument("--output-dir", required=True)

    # Metadata only: single-task and multi-task methods are different workloads.
    p.add_argument("--task-scope", default="")

    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA is required"

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ---------------------------------------------------------
    # Config
    # ---------------------------------------------------------
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    cfg.defrost()

    cfg.MODEL.DEVICE = "cuda"
    cfg.DATASETS.TEST = (args.dataset,)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SEED = args.seed
    cfg.OUTPUT_DIR = args.output_dir

    cfg.freeze()

    if hasattr(cfg, "CUDNN_BENCHMARK"):
        torch.backends.cudnn.benchmark = bool(cfg.CUDNN_BENCHMARK)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    # ---------------------------------------------------------
    # Build exact model through same Trainer path as evaluation.
    # CTCMT_MTL loads student / teacher / anchor internally.
    # ---------------------------------------------------------
    model = Trainer.build_model(cfg)
    model.eval()

    loader = Trainer.build_test_loader(cfg, args.dataset)

    # Force any lazy CUDA initialization before measuring.
    torch.cuda.synchronize()

    # ---------------------------------------------------------
    # Parameter accounting
    # ---------------------------------------------------------
    total_params = count_params(model)
    trainable_params = count_trainable(model)

    # CTCMT_MTL / CoTTA_SemSeg expose student/teacher/anchor. AMROD and TENT wrap
    # their adapted network as ``model`` (AMROD's EMA teacher is ``model_teacher``;
    # its source anchor is a GPU state_dict copy, so it has no module to count).
    student = getattr(model, "student", None)
    if student is None:
        student = getattr(model, "model", None)
    teacher = getattr(model, "teacher", None)
    if teacher is None:
        teacher = getattr(model, "model_teacher", None)
    anchor = getattr(model, "anchor", None)

    student_params = count_params(student)
    teacher_params = count_params(teacher)
    anchor_params = count_params(anchor)

    student_trainable = count_trainable(student)

    resident_model_bytes = module_bytes(model)

    # Useful resolved config provenance.
    solver = cfg.SOLVER

    resolved = {
        "ctcr_weight": float(getattr(solver, "CTCMT_WEIGHT_CTCR", 0.0)),
        "ctcr_mode": str(getattr(solver, "CTCMT_CTCR_MODE", "")),
        "ctcl_weight": float(getattr(solver, "CTCMT_WEIGHT_CTCL", 0.0)),
        "ctpv": bool(getattr(solver, "CTCMT_CTPV_ENABLED", False)),
        "conflict_mode": str(getattr(solver, "CTCMT_CONFLICT_MODE", "")),
        "route_lambda": float(
            getattr(solver, "CTCMT_AUX_TRUNK_LAMBDA", 1.0)
        ),
        "adaptive_routing": bool(
            getattr(solver, "CTCMT_ADAPTIVE_ROUTING", False)
        ),
    }

    print("=" * 78)
    print("PAPER26 EFFICIENCY BENCHMARK")
    print("=" * 78)
    print("method       :", args.method)
    print("config       :", args.config_file)
    print("meta-arch    :", cfg.MODEL.META_ARCHITECTURE)
    print("weights      :", cfg.MODEL.WEIGHTS)
    print("task scope   :", args.task_scope)
    print("dataset      :", args.dataset)
    print("seed/repeat  :", args.seed, args.repeat)
    print("warmup       :", args.warmup)
    print("measure      :", args.measure)
    print("resolved     :", resolved)
    print("GPU          :", torch.cuda.get_device_name(0))
    print("total params :", total_params)
    print("trainable    :", trainable_params)
    print("=" * 78)

    # ---------------------------------------------------------
    # Benchmark
    #
    # IMPORTANT:
    # perf_counter + cuda.synchronize is intentional.
    # CUDA events alone would miss CPU-side work, including the
    # Task-CoCo gradient-consensus optimization.
    #
    # Data loading itself is OUTSIDE the timer.
    # CPU->GPU transfer inside model.forward IS included.
    # ---------------------------------------------------------
    latencies_ms = []
    warmup_seen = 0
    measured_seen = 0

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    with EventStorage(0) as storage:
        for batch in loader:

            n_img = len(batch)
            if n_img != 1:
                raise RuntimeError(
                    f"Expected test batch size 1, got {n_img}"
                )

            # ---------------- warm-up ----------------
            if warmup_seen < args.warmup:
                torch.cuda.synchronize()

                out = model(batch)

                torch.cuda.synchronize()

                del out
                warmup_seen += 1
                storage.step()
                continue

            # Record peak during warm-up / initialization.
            # peak_warmup_alloc = torch.cuda.max_memory_allocated()
            # peak_warmup_res = torch.cuda.max_memory_reserved()

            # Reset peak counter exactly once when timed region begins.
            if measured_seen == 0:
                peak_warmup_alloc = torch.cuda.max_memory_allocated()
                peak_warmup_res = torch.cuda.max_memory_reserved()

                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            # ---------------- measured ----------------
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            out = model(batch)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            dt_ms = (t1 - t0) * 1000.0
            latencies_ms.append(dt_ms)

            del out

            measured_seen += 1
            storage.step()

            if measured_seen >= args.measure:
                break

    if warmup_seen != args.warmup:
        raise RuntimeError(
            f"Only {warmup_seen}/{args.warmup} warmup images available"
        )

    if measured_seen != args.measure:
        raise RuntimeError(
            f"Only {measured_seen}/{args.measure} measured images available"
        )

    torch.cuda.synchronize()

    peak_measure_alloc = torch.cuda.max_memory_allocated()
    peak_measure_res = torch.cuda.max_memory_reserved()

    peak_alloc = max(peak_warmup_alloc, peak_measure_alloc)
    peak_res = max(peak_warmup_res, peak_measure_res)

    mean_ms = statistics.mean(latencies_ms)
    std_ms = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    median_ms = statistics.median(latencies_ms)

    fps = 1000.0 / mean_ms

    result = {
        "method": args.method,
        "task_scope": args.task_scope,
        "meta_architecture": cfg.MODEL.META_ARCHITECTURE,
        "model_weights": cfg.MODEL.WEIGHTS,
        "config_file": args.config_file,
        "dataset": args.dataset,
        "seed": args.seed,
        "repeat": args.repeat,

        "warmup_images": args.warmup,
        "measured_images": args.measure,

        "mean_ms_per_image": mean_ms,
        "std_ms_per_image": std_ms,
        "median_ms_per_image": median_ms,
        "p95_ms_per_image": percentile(latencies_ms, 95),
        "min_ms_per_image": min(latencies_ms),
        "max_ms_per_image": max(latencies_ms),
        "fps_from_mean": fps,

        "peak_allocated_mib": mib(peak_alloc),
        "peak_reserved_mib": mib(peak_res),

        "peak_warmup_allocated_mib": mib(peak_warmup_alloc),
        "peak_measured_allocated_mib": mib(peak_measure_alloc),

        "total_params": total_params,
        "trainable_params": trainable_params,

        "student_params": student_params,
        "student_trainable_params": student_trainable,
        "teacher_params": teacher_params,
        "anchor_params": anchor_params,

        "resident_model_mib": mib(resident_model_bytes),

        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),

        "resolved_config": resolved,

        # Keep raw samples for later statistical checks.
        "latencies_ms": latencies_ms,
    }

    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)

    print()
    print("=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"mean latency       : {mean_ms:.3f} ms/image")
    print(f"median latency     : {median_ms:.3f} ms/image")
    print(f"p95 latency        : {result['p95_ms_per_image']:.3f} ms/image")
    print(f"FPS                : {fps:.3f}")
    print(f"peak allocated     : {result['peak_allocated_mib']:.1f} MiB")
    print(f"peak reserved      : {result['peak_reserved_mib']:.1f} MiB")
    print(f"total params       : {total_params:,}")
    print(f"trainable params   : {trainable_params:,}")
    print(f"student params     : {student_params:,}")
    print(f"teacher params     : {teacher_params:,}")
    print(f"anchor params      : {anchor_params:,}")
    print(f"resident model     : {result['resident_model_mib']:.1f} MiB")
    print()
    print("JSON:", args.output_json)


if __name__ == "__main__":
    main()
