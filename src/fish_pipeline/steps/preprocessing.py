"""Image pre-processing implemented without external dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


Matrix = List[List[float]]


def _ensure_grayscale(image: List[List[float]] | List[List[List[float]]]) -> Matrix:
    if not image:
        return []
    first_row = image[0]
    if first_row and isinstance(first_row[0], list):
        # Convert RGB to grayscale via average
        grayscale: Matrix = []
        for row in image:  # type: ignore[arg-type]
            grayscale.append([sum(pixel) / len(pixel) for pixel in row])
        return grayscale
    return [list(row) for row in image]  # type: ignore[arg-type]


def _mean_filter(matrix: Matrix, kernel_size: int = 3) -> Matrix:
    if not matrix:
        return []
    height = len(matrix)
    width = len(matrix[0])
    radius = kernel_size // 2
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


def _normalize(matrix: Matrix) -> Matrix:
    if not matrix:
        return []
    flat = [value for row in matrix for value in row]
    min_val = min(flat)
    max_val = max(flat)
    if max_val == min_val:
        return [[0.0 for _ in row] for row in matrix]
    return [[(value - min_val) / (max_val - min_val) for value in row] for row in matrix]


def _resize(matrix: Matrix, target_size: Tuple[int, int]) -> Matrix:
    if not matrix:
        return []
    target_h, target_w = target_size
    height = len(matrix)
    width = len(matrix[0])
    result = [[0.0 for _ in range(target_w)] for _ in range(target_h)]
    for y in range(target_h):
        src_y = int(round((y / max(target_h - 1, 1)) * (height - 1))) if target_h > 1 else 0
        for x in range(target_w):
            src_x = int(round((x / max(target_w - 1, 1)) * (width - 1))) if target_w > 1 else 0
            result[y][x] = matrix[src_y][src_x]
    return result


@dataclass
class PreprocessingConfig:
    target_size: Optional[Tuple[int, int]] = None
    kernel_size: int = 3
    normalize: bool = True


class Preprocessor:
    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config

    def run(
        self,
        image: List[List[float]] | List[List[List[float]]],
        depth: Matrix,
    ) -> Tuple[Matrix, Matrix]:
        grayscale = _ensure_grayscale(image)
        smoothed = _mean_filter(grayscale, kernel_size=self.config.kernel_size)
        if self.config.normalize:
            smoothed = _normalize(smoothed)

        depth_smoothed = _mean_filter(depth, kernel_size=self.config.kernel_size)

        if self.config.target_size:
            smoothed = _resize(smoothed, self.config.target_size)
            depth_smoothed = _resize(depth_smoothed, self.config.target_size)

        return smoothed, depth_smoothed
