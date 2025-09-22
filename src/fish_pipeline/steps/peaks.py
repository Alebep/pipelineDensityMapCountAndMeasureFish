"""Peak detection for density maps without external dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


Matrix = List[List[float]]


def _mean_filter(matrix: Matrix, radius: int) -> Matrix:
    if not matrix:
        return []
    height = len(matrix)
    width = len(matrix[0])
    result = [[0.0 for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            values = []
            for ky in range(-radius, radius + 1):
                for kx in range(-radius, radius + 1):
                    ny = min(max(y + ky, 0), height - 1)
                    nx = min(max(x + kx, 0), width - 1)
                    values.append(matrix[ny][nx])
            result[y][x] = sum(values) / len(values)
    return result


def _neighbourhood(matrix: Matrix, x: int, y: int, radius: int) -> Sequence[float]:
    height = len(matrix)
    width = len(matrix[0]) if height else 0
    values = []
    for ky in range(-radius, radius + 1):
        for kx in range(-radius, radius + 1):
            ny = min(max(y + ky, 0), height - 1)
            nx = min(max(x + kx, 0), width - 1)
            values.append(matrix[ny][nx])
    return values


@dataclass
class Peak:
    x: int
    y: int
    value: float


class PeakDetector:
    def __init__(self, radius: int = 2, threshold: float = 0.1) -> None:
        self.radius = radius
        self.threshold = threshold

    def run(self, density_map: Matrix) -> List[Peak]:
        smoothed = _mean_filter(density_map, radius=self.radius)
        height = len(smoothed)
        width = len(smoothed[0]) if height else 0
        peaks: List[Peak] = []
        for y in range(height):
            for x in range(width):
                value = smoothed[y][x]
                if value <= self.threshold:
                    continue
                neighbours = _neighbourhood(smoothed, x, y, self.radius)
                if value >= max(neighbours):
                    peaks.append(Peak(x=x, y=y, value=value))
        peaks.sort(key=lambda peak: peak.value, reverse=True)
        return peaks
