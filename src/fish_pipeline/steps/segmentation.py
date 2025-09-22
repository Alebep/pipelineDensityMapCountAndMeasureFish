"""Instance segmentation via promptable model (pure Python)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from .peaks import Peak

Mask = List[List[bool]]


def _create_empty_mask(height: int, width: int) -> Mask:
    return [[False for _ in range(width)] for _ in range(height)]


def _circle_mask(center_x: int, center_y: int, radius: int, height: int, width: int) -> Mask:
    mask = _create_empty_mask(height, width)
    radius_sq = radius * radius
    for y in range(max(center_y - radius, 0), min(center_y + radius + 1, height)):
        for x in range(max(center_x - radius, 0), min(center_x + radius + 1, width)):
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius_sq:
                mask[y][x] = True
    return mask


def _mask_score(mask: Mask) -> float:
    total = sum(1 for row in mask for value in row if value)
    return min(max(total / 400.0, 0.5), 1.0)


@dataclass
class MaskInstance:
    mask: Mask
    center: Tuple[int, int]
    prompt_type: str
    score: float


class PromptedSegmenter:
    def __init__(self, radius: int = 8, model_name: str = "SAM-HQ", model_version: str = "0.4.1") -> None:
        self.radius = radius
        self.model_name = model_name
        self.model_version = model_version

    def run(self, image: List[List[float]], peaks: Iterable[Peak]) -> List[MaskInstance]:
        height = len(image)
        width = len(image[0]) if height else 0
        instances: List[MaskInstance] = []
        for peak in peaks:
            mask = _circle_mask(peak.x, peak.y, self.radius, height, width)
            score = _mask_score(mask)
            instances.append(
                MaskInstance(
                    mask=mask,
                    center=(peak.x, peak.y),
                    prompt_type="point",
                    score=score,
                )
            )
        return instances

    def metadata(self) -> Tuple[str, str]:
        return self.model_name, self.model_version
