# Copyright (c) OpenMMLab. All rights reserved.
"""Per-domain bbox metric for a stream of concatenated domains.

``CityscapesMetricNew`` slices the results of a ConcatDataset with a hard-coded
``4 x 500``.  This variant takes the real domain sizes, so it works for ACDC (400
images per condition) and Cityscapes-C (500 images per corruption, 12 conditions).

It evaluates every domain separately (image ids are only unique inside one domain
json, so the whole stream is NOT evaluated at once) and logs, per domain, one
line that is easy to parse::

    [DOMAIN i/n] AP=... AP50=... AP75=...

Use together with an unmodified detector (source-only) or with any adapter that
returns ``pred_det_instances``.
"""
from typing import Any, Dict, List, Optional, Sequence

from mmdet.evaluation import CocoMetric
from mmengine.dist import broadcast_object_list, collect_results, is_main_process
from mmengine.logging import print_log

from shift_tta.registry import METRICS
from .cityscapes_metric import CityscapesMetricNew, _to_cpu


@METRICS.register_module()
class CocoMetricDomains(CityscapesMetricNew):
    """Args:
        domain_sizes (list[int]): number of images of each domain, in stream order.
        domain_names (list[str], optional): names used in the log lines.
    """

    def __init__(self,
                 domain_sizes: Sequence[int],
                 domain_names: Optional[Sequence[str]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.domain_sizes = [int(s) for s in domain_sizes]
        if domain_names is not None:
            assert len(domain_names) == len(self.domain_sizes)
        self.domain_names = list(domain_names) if domain_names is not None else None

    def evaluate(self, size: int) -> dict:
        assert size == sum(self.domain_sizes), (
            f'dataset size {size} != sum(domain_sizes) {sum(self.domain_sizes)}')
        if self.collect_device == 'cpu':
            results = collect_results(
                self.results, size, self.collect_device, tmpdir=self.collect_dir)
        else:
            results = collect_results(self.results, size, self.collect_device)

        if is_main_process():
            results = _to_cpu(results)
            per_domain: List[Dict[str, float]] = []
            start = 0
            for i, n in enumerate(self.domain_sizes):
                chunk = results[start:start + n]
                start += n
                # CocoMetric caches these on first use; image ids differ between domain jsons
                self._coco_api = None
                self.img_ids = None
                self.cat_ids = None
                m = self.compute_metrics(chunk)
                # a stream whose predictions are all invalid (e.g. diverged weights) yields no AP keys
                for k in ('bbox_mAP', 'bbox_mAP_50', 'bbox_mAP_75'):
                    m.setdefault(k, float('nan'))
                per_domain.append(m)
                name = self.domain_names[i] if self.domain_names else str(i)
                print_log(
                    f"[DOMAIN {i + 1}/{len(self.domain_sizes)}] name={name} n={n} "
                    f"AP={m['bbox_mAP']:.4f} AP50={m['bbox_mAP_50']:.4f} "
                    f"AP75={m['bbox_mAP_75']:.4f}",
                    logger='current')
            self._coco_api = None
            mean = {k: sum(m[k] for m in per_domain) / len(per_domain)
                    for k in ('bbox_mAP', 'bbox_mAP_50', 'bbox_mAP_75')}
            print_log(
                f"[MEAN over {len(per_domain)} domains] AP={mean['bbox_mAP']:.4f} "
                f"AP50={mean['bbox_mAP_50']:.4f}", logger='current')
            if self.prefix:
                mean = {'/'.join((self.prefix, k)): v for k, v in mean.items()}
            metrics = [mean]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)
        self.results.clear()
        return metrics[0]
