"""Per-domain mIoU for a stream of concatenated domains.

Same protocol as the AMROD benchmark tables: the confusion matrix is accumulated
inside each domain, mIoU is computed per domain, and the reported number is the
mean over domains. One parsable line per domain::

    [SEGDOMAIN i/n] name=... n=... mIoU=...
"""
from typing import List, Optional, Sequence

import numpy as np
from mmengine.dist import broadcast_object_list, collect_results, is_main_process
from mmengine.logging import print_log

from shift_tta.registry import METRICS
from .iou_metric import IoUMetric


def _miou(bins: np.ndarray, k: int) -> float:
    total = bins.reshape(k, k).astype(np.float64)
    tp = np.diag(total)
    denom = total.sum(0) + total.sum(1) - tp
    iou = np.where(denom > 0, tp / np.maximum(denom, 1e-12), np.nan)
    return float(np.nanmean(iou))


@METRICS.register_module()
class IoUMetricDomains(IoUMetric):

    def __init__(self, domain_sizes: Sequence[int],
                 domain_names: Optional[Sequence[str]] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.domain_sizes = [int(s) for s in domain_sizes]
        self.domain_names = list(domain_names) if domain_names is not None else None

    def process(self, data_batch, data_samples) -> None:
        # one entry per sample, even when a sample carries no prediction
        n0 = len(self.results)
        super().process(data_batch, data_samples)
        assert len(self.results) - n0 == len(data_samples), 'sample without pred/gt sem_seg'

    def evaluate(self, size: int) -> dict:
        assert size == sum(self.domain_sizes), (size, sum(self.domain_sizes))
        results = collect_results(self.results, size, self.collect_device)
        if is_main_process():
            vals: List[float] = []
            start = 0
            for i, n in enumerate(self.domain_sizes):
                bins = np.sum(np.stack(results[start:start + n]), axis=0)
                start += n
                v = _miou(bins, self.num_classes)
                vals.append(v)
                name = self.domain_names[i] if self.domain_names else str(i)
                print_log(f'[SEGDOMAIN {i + 1}/{len(self.domain_sizes)}] name={name} n={n} '
                          f'mIoU={v:.4f}', logger='current')
            mean = float(np.mean(vals))
            print_log(f'[SEGMEAN over {len(vals)} domains] mIoU={mean:.4f}', logger='current')
            metrics = [{'mIoU': mean}]
        else:
            metrics = [None]
        broadcast_object_list(metrics)
        self.results.clear()
        return metrics[0]
