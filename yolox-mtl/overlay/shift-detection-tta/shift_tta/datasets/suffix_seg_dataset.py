"""CocoDataset whose semantic-segmentation path strips the RGB suffix first.

mmdet builds ``seg_map_path`` as ``<seg prefix>/<file_name_no_ext><seg_map_suffix>``.
ACDC images end with ``_rgb_anon`` and their masks with ``_gt_labelTrainIds``, so
the RGB suffix has to be removed before the mask suffix is appended.
"""
from __future__ import annotations

from typing import Optional

from mmdet.datasets import CocoDataset

from shift_tta.registry import DATASETS


@DATASETS.register_module()
class SuffixSegDataset(CocoDataset):
    """Args:
        rgb_suffix: trailing token of the RGB file stem to drop (e.g. ``_rgb_anon``).
    """

    def __init__(self, *args, rgb_suffix: str = '', **kwargs) -> None:
        self.rgb_suffix = rgb_suffix
        super().__init__(*args, **kwargs)

    def parse_data_info(self, raw_data_info: dict) -> Optional[dict]:
        data_info = super().parse_data_info(raw_data_info)
        if data_info is None:
            return None
        seg_map_path = data_info.get('seg_map_path')
        if seg_map_path and self.rgb_suffix:
            old = self.rgb_suffix + self.seg_map_suffix
            assert old in seg_map_path, seg_map_path
            data_info['seg_map_path'] = seg_map_path.replace(old, self.seg_map_suffix)
        return data_info
