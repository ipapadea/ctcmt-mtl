"""Cityscapes dataset variant that builds the correct seg_map_path.

mmdet's :class:`CocoDataset.parse_data_info` builds ``seg_map_path`` as::

    <data_prefix.seg>/<file_name_no_ext><seg_map_suffix>

For Cityscapes the RGB file names end with ``_leftImg8bit.png`` while the
semseg files are named ``*_gtFine_labelTrainIds.png`` — i.e. the trailing
``_leftImg8bit`` has to be stripped before the seg suffix is appended.
This subclass does exactly that.
"""

from __future__ import annotations

from typing import Optional

from mmdet.datasets import CityscapesDataset as _MMDetCityscapesDataset

from shift_tta.registry import DATASETS


@DATASETS.register_module()
class CityscapesSegDataset(_MMDetCityscapesDataset):
    """Cityscapes dataset with correct semseg-path construction."""

    _RGB_SUFFIX = "_leftImg8bit"

    def parse_data_info(self, raw_data_info: dict) -> Optional[dict]:
        data_info = super().parse_data_info(raw_data_info)
        if data_info is None:
            return None
        seg_map_path = data_info.get("seg_map_path")
        if seg_map_path and (self._RGB_SUFFIX + self.seg_map_suffix) in seg_map_path:
            data_info["seg_map_path"] = seg_map_path.replace(
                self._RGB_SUFFIX + self.seg_map_suffix,
                self.seg_map_suffix,
            )
        return data_info
