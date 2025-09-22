"""Calibration helpers implemented using the standard library."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import List, Sequence, Tuple

from ..data.entities import CalibrationParameters


DEPTH_UNIT_TO_MM = {
    "mm": 1.0,
    "m": 1000.0,
    "cm": 10.0,
}


@dataclass
class DepthProjector:
    calibration: CalibrationParameters

    def depth_scale(self, depth_units: str) -> float:
        return DEPTH_UNIT_TO_MM.get(depth_units.lower(), 1.0)

    def rgb_to_depth_coordinates(self, x: float, y: float) -> Tuple[float, float]:
        ratios = self.calibration.alignment
        if ratios.type != "ratios":
            return x, y
        return x / ratios.ratioW, y / ratios.ratioH

    def pixel_to_camera(self, x: float, y: float, depth_value: float, depth_units: str) -> Tuple[float, float, float]:
        scale = self.depth_scale(depth_units)
        z = depth_value * scale
        fx = self.calibration.fx
        fy = self.calibration.fy
        cx = self.calibration.cx
        cy = self.calibration.cy
        X = (x - cx) * z / fx
        Y = (y - cy) * z / fy
        return float(X), float(Y), float(z)

    def polyline_length_mm(
        self,
        polyline: Sequence[Tuple[float, float]],
        depth_map: List[List[float]],
        depth_units: str,
    ) -> Tuple[float, List[Tuple[float, float, float]], float, int]:
        if len(polyline) < 2:
            return 0.0, [], 0.0, 0

        projected_points: List[Tuple[float, float, float]] = []
        valid_segments = 0
        total_length = 0.0

        height = len(depth_map)
        width = len(depth_map[0]) if height else 0

        for x, y in polyline:
            depth_x, depth_y = self.rgb_to_depth_coordinates(x, y)
            depth_x = max(0, min(width - 1, int(round(depth_x))))
            depth_y = max(0, min(height - 1, int(round(depth_y))))
            depth_value = depth_map[depth_y][depth_x]
            if depth_value <= 0:
                projected_points.append((float("nan"), float("nan"), float("nan")))
            else:
                projected_points.append(self.pixel_to_camera(x, y, depth_value, depth_units))

        for idx in range(1, len(projected_points)):
            p1 = projected_points[idx - 1]
            p2 = projected_points[idx]
            if any(value != value for value in p1) or any(value != value for value in p2):  # check for NaN
                continue
            distance = sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
            total_length += distance
            valid_segments += 1

        valid_fraction = valid_segments / max(len(polyline) - 1, 1)
        return total_length, projected_points, valid_fraction, valid_segments
