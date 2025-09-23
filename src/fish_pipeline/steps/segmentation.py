"""Instance segmentation powered by promptable models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
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


def _mask_area(mask: Mask) -> int:
    return sum(1 for row in mask for value in row if value)


def _mean_intensity(mask: Mask, image: List[List[float]]) -> float:
    total = 0.0
    count = 0
    for y, row in enumerate(mask):
        for x, is_foreground in enumerate(row):
            if is_foreground:
                total += image[y][x]
                count += 1
    return total / count if count else 0.0


def _within_bounds(x: int, y: int, width: int, height: int) -> bool:
    return 0 <= x < width and 0 <= y < height


@dataclass
class MaskInstance:
    mask: Mask
    center: Tuple[int, int]
    prompt_type: str
    score: float


class BasePromptedSegmenter(ABC):
    """Abstract base class for promptable segmenters."""

    model_name: str = "prompted-segmenter"
    model_version: str = "0.0.0"

    @abstractmethod
    def run(self, image: List[List[float]], peaks: Iterable[Peak]) -> List[MaskInstance]:
        """Produce segmentation masks for each peak."""

    def metadata(self) -> Tuple[str, str]:
        return self.model_name, self.model_version


class SAMHQSegmenter(BasePromptedSegmenter):
    """Simplified SAM-HQ style promptable segmenter."""

    def __init__(
        self,
        intensity_ratio: float = 0.45,
        max_radius: int = 18,
        dilation_radius: int = 2,
        min_area: int = 24,
        min_radius: int = 6,
        model_version: str = "0.5.0",
    ) -> None:
        self.intensity_ratio = intensity_ratio
        self.max_radius = max_radius
        self.dilation_radius = dilation_radius
        self.min_area = min_area
        self.min_radius = min_radius
        self.model_name = "SAM-HQ"
        self.model_version = model_version
        self._score_normalizer = float(max_radius * max_radius)

    def run(self, image: List[List[float]], peaks: Iterable[Peak]) -> List[MaskInstance]:
        height = len(image)
        width = len(image[0]) if height else 0
        if height == 0 or width == 0:
            return []
        global_max = max((max(row) for row in image), default=1e-6)
        instances: List[MaskInstance] = []
        for peak in peaks:
            mask = self._segment_single(image, peak, width, height, global_max)
            score = self._score(mask, image, global_max, peak.value)
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

    def _segment_single(
        self,
        image: List[List[float]],
        peak: Peak,
        width: int,
        height: int,
        global_max: float,
    ) -> Mask:
        adaptive_threshold = self._adaptive_threshold(image, peak, width, height, global_max)
        region = self._region_grow(image, peak, width, height, adaptive_threshold)
        if self.dilation_radius > 0:
            region = self._dilate(region, width, height, self.dilation_radius)
        area = _mask_area(region)
        if area < self.min_area:
            radius = max(self.min_radius, self.max_radius // 2)
            return _circle_mask(peak.x, peak.y, radius, height, width)
        return region

    def _adaptive_threshold(
        self,
        image: List[List[float]],
        peak: Peak,
        width: int,
        height: int,
        global_max: float,
    ) -> float:
        local_values = []
        radius = 3
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = peak.x + dx, peak.y + dy
                if _within_bounds(nx, ny, width, height):
                    local_values.append(image[ny][nx])
        if not local_values:
            return max(peak.value * self.intensity_ratio, 0.0)
        local_values.sort()
        trim = max(len(local_values) // 6, 1)
        trimmed = local_values[trim:-trim] if len(local_values) > 2 * trim else local_values
        baseline = sum(trimmed) / len(trimmed)
        contrast = peak.value - baseline
        adaptive = baseline + contrast * self.intensity_ratio
        adaptive = max(min(adaptive, peak.value), 0.0)
        if global_max > 0:
            adaptive = max(adaptive, 0.05 * global_max)
        return adaptive

    def _region_grow(
        self,
        image: List[List[float]],
        peak: Peak,
        width: int,
        height: int,
        threshold: float,
    ) -> Mask:
        mask = _create_empty_mask(height, width)
        queue: deque[Tuple[int, int]] = deque()
        queue.append((peak.x, peak.y))
        visited = set()
        max_distance_sq = self.max_radius * self.max_radius
        while queue:
            x, y = queue.popleft()
            if (x, y) in visited:
                continue
            visited.add((x, y))
            if not _within_bounds(x, y, width, height):
                continue
            if (x - peak.x) ** 2 + (y - peak.y) ** 2 > max_distance_sq:
                continue
            value = image[y][x]
            if value < threshold:
                continue
            mask[y][x] = True
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if (nx, ny) not in visited:
                        queue.append((nx, ny))
        return mask

    def _dilate(self, mask: Mask, width: int, height: int, radius: int) -> Mask:
        if radius <= 0:
            return mask
        dilated = _create_empty_mask(height, width)
        for y in range(height):
            for x in range(width):
                if not mask[y][x]:
                    continue
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        nx, ny = x + dx, y + dy
                        if _within_bounds(nx, ny, width, height):
                            dilated[ny][nx] = True
        return dilated

    def _score(self, mask: Mask, image: List[List[float]], global_max: float, peak_value: float) -> float:
        area = _mask_area(mask)
        if area == 0:
            return 0.0
        mean_intensity = _mean_intensity(mask, image)
        intensity_factor = mean_intensity / (global_max + 1e-6)
        area_factor = min(area / (self._score_normalizer + 1e-6), 1.0)
        peak_factor = peak_value / (global_max + 1e-6)
        score = 0.25 + 0.45 * intensity_factor + 0.3 * ((area_factor + peak_factor) / 2.0)
        return max(0.0, min(score, 1.0))


PromptedSegmenter = SAMHQSegmenter
