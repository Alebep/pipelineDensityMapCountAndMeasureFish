"""Density map model definitions without external dependencies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Tuple

Matrix = List[List[float]]
TrainingSample = Tuple[Matrix, Matrix]


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


def _matrix_sum(matrix: Matrix) -> float:
    return sum(sum(row) for row in matrix)


class BaseDensityModel(ABC):
    name: str = "BaseDensityModel"
    version: str = "0.0.1"

    @abstractmethod
    def train(self, dataset: Iterable[TrainingSample]) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: Matrix) -> Matrix:
        raise NotImplementedError


@dataclass
class LFCNetLikeDensityModel(BaseDensityModel):
    kernel_size: int = 3

    def __post_init__(self) -> None:
        self.name = "LFCNet"
        self.version = "1.2.0"
        self._scale = 1.0

    def train(self, dataset: Iterable[TrainingSample]) -> None:
        total_target = 0.0
        total_prediction = 0.0
        count = 0
        for image, target in dataset:
            prediction = self._feature_map(image)
            total_target += _matrix_sum(target)
            total_prediction += _matrix_sum(prediction)
            count += 1
        if count == 0:
            raise ValueError("Dataset cannot be empty")
        if total_prediction > 0:
            self._scale = total_target / total_prediction
        else:
            self._scale = 1.0

    def _feature_map(self, image: Matrix) -> Matrix:
        smoothed = _mean_filter(image, kernel_size=self.kernel_size)
        return _normalize(smoothed)

    def predict(self, image: Matrix) -> Matrix:
        feature = self._feature_map(image)
        return [[value * self._scale for value in row] for row in feature]

    def metadata(self) -> Tuple[str, str]:
        return self.name, self.version
