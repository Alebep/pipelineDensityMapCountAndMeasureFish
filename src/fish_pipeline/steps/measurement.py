"""Length measurement strategies implemented in pure Python."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot
from typing import Dict, List, Optional, Tuple

from ..utils.calibration import DepthProjector
from ..utils.geometry import (
    contour_from_mask,
    farthest_point_from_reference,
    polyline_length,
    raster_to_points,
)

Mask = List[List[bool]]


@dataclass
class MeasurementComputation:
    path_rgb: List[Tuple[float, float]]
    length_px: float
    length_mm: float
    head: Tuple[float, float]
    tail: Tuple[float, float]
    method: str
    depth_points: List[Tuple[float, float, float]]
    depth_valid_fraction: float
    segments_used: int


class LengthMeasurementStrategy:
    method_name: str = "base"

    def measure(
        self,
        mask: Mask,
        head_point: Tuple[int, int],
        depth_map: List[List[float]],
        depth_units: str,
        projector: DepthProjector,
    ) -> MeasurementComputation:
        raise NotImplementedError


class EndToEndMeasurementStrategy(LengthMeasurementStrategy):
    method_name = "end_to_end"

    def measure(
        self,
        mask: Mask,
        head_point: Tuple[int, int],
        depth_map: List[List[float]],
        depth_units: str,
        projector: DepthProjector,
    ) -> MeasurementComputation:
        contour = contour_from_mask(mask)
        if not contour:
            raise ValueError("Mask has no contour points")
        tail_point, _, distance = farthest_point_from_reference(contour, head_point)
        path = [tuple(map(float, head_point)), tuple(map(float, tail_point))]
        length_mm, depth_points, valid_fraction, segments_used = projector.polyline_length_mm(
            path, depth_map, depth_units
        )
        return MeasurementComputation(
            path_rgb=path,
            length_px=distance,
            length_mm=length_mm,
            head=tuple(map(float, head_point)),
            tail=tuple(map(float, tail_point)),
            method=self.method_name,
            depth_points=depth_points,
            depth_valid_fraction=valid_fraction,
            segments_used=segments_used,
        )


class SkeletonMeasurementStrategy(LengthMeasurementStrategy):
    method_name = "skeleton"

    def __init__(self, max_nodes: int = 5000) -> None:
        self.max_nodes = max_nodes

    def measure(
        self,
        mask: Mask,
        head_point: Tuple[int, int],
        depth_map: List[List[float]],
        depth_units: str,
        projector: DepthProjector,
    ) -> MeasurementComputation:
        head = self._closest_point_on_mask(mask, head_point)
        path_pixels = self._longest_path(mask, head)
        if len(path_pixels) < 2:
            raise ValueError("Unable to compute skeleton path")
        path_float = [(float(x), float(y)) for x, y in path_pixels]
        length_px = polyline_length(path_float)
        length_mm, depth_points, valid_fraction, segments_used = projector.polyline_length_mm(
            path_float, depth_map, depth_units
        )
        return MeasurementComputation(
            path_rgb=path_float,
            length_px=length_px,
            length_mm=length_mm,
            head=(float(head[0]), float(head[1])),
            tail=(float(path_pixels[-1][0]), float(path_pixels[-1][1])),
            method=self.method_name,
            depth_points=depth_points,
            depth_valid_fraction=valid_fraction,
            segments_used=segments_used,
        )

    def _closest_point_on_mask(self, mask: Mask, point: Tuple[int, int]) -> Tuple[int, int]:
        if self._is_inside(mask, point):
            return point
        points = raster_to_points(mask)
        if not points:
            raise ValueError("Empty mask")
        best_point = points[0]
        best_dist = hypot(points[0][0] - point[0], points[0][1] - point[1])
        for candidate in points[1:]:
            dist = hypot(candidate[0] - point[0], candidate[1] - point[1])
            if dist < best_dist:
                best_dist = dist
                best_point = candidate
        return best_point

    def _is_inside(self, mask: Mask, point: Tuple[int, int]) -> bool:
        x, y = point
        height = len(mask)
        width = len(mask[0]) if height else 0
        if x < 0 or y < 0 or y >= height or x >= width:
            return False
        return mask[y][x]

    def _longest_path(self, mask: Mask, head: Tuple[int, int]) -> List[Tuple[int, int]]:
        height = len(mask)
        width = len(mask[0]) if height else 0
        visited = [[False for _ in range(width)] for _ in range(height)]
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
        distance = [[-1 for _ in range(width)] for _ in range(height)]

        queue: deque[Tuple[int, int]] = deque()
        queue.append(head)
        visited[head[1]][head[0]] = True
        distance[head[1]][head[0]] = 0
        parent[head] = None
        farthest = head

        while queue and len(parent) < self.max_nodes:
            cx, cy = queue.popleft()
            for nx, ny in self._neighbours(cx, cy, width, height):
                if not mask[ny][nx] or visited[ny][nx]:
                    continue
                visited[ny][nx] = True
                parent[(nx, ny)] = (cx, cy)
                distance[ny][nx] = distance[cy][cx] + 1
                queue.append((nx, ny))
                if distance[ny][nx] > distance[farthest[1]][farthest[0]]:
                    farthest = (nx, ny)

        path: List[Tuple[int, int]] = []
        current: Optional[Tuple[int, int]] = farthest
        while current is not None:
            path.append(current)
            current = parent.get(current)
        return list(reversed(path))

    def _neighbours(self, x: int, y: int, width: int, height: int) -> List[Tuple[int, int]]:
        candidates = [
            (x - 1, y),
            (x + 1, y),
            (x, y - 1),
            (x, y + 1),
        ]
        return [c for c in candidates if 0 <= c[0] < width and 0 <= c[1] < height]
