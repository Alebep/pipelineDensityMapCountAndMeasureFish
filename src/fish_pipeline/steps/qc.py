"""Quality control for segmentation masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..data.entities import MaskStats, QualityControl
from ..utils.geometry import bounding_box_from_mask


@dataclass
class QCEvaluation:
    mask_stats: MaskStats
    qc: QualityControl
    errors: List[str]


class QualityController:
    def __init__(self, min_area: int = 100, border_margin: int = 1) -> None:
        self.min_area = min_area
        self.border_margin = border_margin

    def evaluate(
        self,
        mask: List[List[bool]],
        prompt_type: str,
        depth_valid_fraction: float,
        method_used: str,
        confidence: float,
    ) -> QCEvaluation:
        area = sum(1 for row in mask for value in row if value)
        x, y, w, h = bounding_box_from_mask(mask)
        touches_border = self._touches_border(mask)
        mask_ok = area >= self.min_area and not touches_border
        occlusion_flag = depth_valid_fraction < 0.7

        mask_stats = MaskStats(
            area_px=area,
            bbox={"x": x, "y": y, "w": w, "h": h},
            touches_border=touches_border,
            iou_suppressed=False,
        )

        qc = QualityControl(
            mask_ok=mask_ok,
            depth_valid_fraction=float(depth_valid_fraction),
            occlusion_flag=occlusion_flag,
            method_used=method_used,
            confidence=float(confidence),
        )

        errors: List[str] = []
        if not mask_ok:
            errors.append("mask_failed_qc")
        if occlusion_flag:
            errors.append("occlusion_suspected")
        if depth_valid_fraction == 0:
            errors.append("no_valid_depth")

        return QCEvaluation(mask_stats=mask_stats, qc=qc, errors=errors)

    def _touches_border(self, mask: List[List[bool]]) -> bool:
        if not mask:
            return False
        height = len(mask)
        width = len(mask[0])
        if any(mask[0]):
            return True
        if any(mask[-1]):
            return True
        for row in mask:
            if row[0] or row[-1]:
                return True
        return False
