"""Density map generation step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..models.density import BaseDensityModel


@dataclass
class DensityMapResult:
    density_map: List[List[float]]


class DensityMapGenerator:
    def __init__(self, model: BaseDensityModel) -> None:
        self.model = model

    def run(self, image: List[List[float]]) -> DensityMapResult:
        density_map = self.model.predict(image)
        return DensityMapResult(density_map=density_map)
