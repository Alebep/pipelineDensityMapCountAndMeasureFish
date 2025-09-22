"""Geometry helper functions implemented with standard Python."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable, List, Sequence, Tuple


@dataclass
class OrientedBBox:
    centroid: Tuple[float, float]
    size: Tuple[float, float]
    angle: float


def oriented_bbox_with_ratio(
    points: Sequence[Tuple[float, float]], ratio: float = 3.0
) -> Tuple[OrientedBBox, List[Tuple[float, float]]]:
    if len(points) < 1:
        raise ValueError("At least one point required")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = max(x_max - x_min, 1.0)
    height = max(y_max - y_min, 1.0)

    current_ratio = max(width, height) / max(min(width, height), 1.0)
    if current_ratio < ratio:
        if width > height:
            height = width / ratio
        else:
            width = height / ratio

    centroid = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
    corners = [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
    ]
    return OrientedBBox(centroid=centroid, size=(width, height), angle=0.0), corners


def ensure_clockwise_quadrilateral(
    x_min: float, y_min: float, x_max: float, y_max: float
) -> List[Tuple[float, float]]:
    return [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
    ]


def polygon_area(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx in range(len(points)):
        x1, y1 = points[idx]
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - y1 * x2
    return 0.5 * area


def bounding_box_from_mask(mask: Sequence[Sequence[bool]]) -> Tuple[int, int, int, int]:
    ys = []
    xs = []
    for y, row in enumerate(mask):
        for x, value in enumerate(row):
            if value:
                xs.append(x)
                ys.append(y)
    if not xs:
        return 0, 0, 0, 0
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    return x_min, y_min, x_max - x_min + 1, y_max - y_min + 1


def contour_from_mask(mask: Sequence[Sequence[bool]]) -> List[Tuple[int, int]]:
    points: List[Tuple[int, int]] = []
    height = len(mask)
    width = len(mask[0]) if height else 0
    for y in range(height):
        for x in range(width):
            if not mask[y][x]:
                continue
            neighbours = [
                (x - 1, y),
                (x + 1, y),
                (x, y - 1),
                (x, y + 1),
            ]
            if any(
                nx < 0
                or ny < 0
                or nx >= width
                or ny >= height
                or not mask[ny][nx]
                for nx, ny in neighbours
            ):
                points.append((x, y))
    return points


def farthest_point_from_reference(
    points: Iterable[Tuple[int, int]], reference: Tuple[int, int]
) -> Tuple[Tuple[int, int], float, float]:
    ref_x, ref_y = reference
    max_dist = -1.0
    farthest = reference
    for x, y in points:
        dist = hypot(x - ref_x, y - ref_y)
        if dist > max_dist:
            max_dist = dist
            farthest = (x, y)
    return farthest, max_dist, max_dist


def polyline_length(points: Sequence[Tuple[float, float]]) -> float:
    length = 0.0
    for idx in range(1, len(points)):
        x1, y1 = points[idx - 1]
        x2, y2 = points[idx]
        length += hypot(x2 - x1, y2 - y1)
    return length


def decimate_polyline(points: Sequence[Tuple[float, float]], step: int) -> List[Tuple[float, float]]:
    if len(points) <= 2 or step <= 1:
        return list(points)
    decimated = [points[0]]
    for idx in range(1, len(points) - 1, step):
        decimated.append(points[idx])
    decimated.append(points[-1])
    return decimated


def raster_to_points(mask: Sequence[Sequence[bool]]) -> List[Tuple[int, int]]:
    points: List[Tuple[int, int]] = []
    for y, row in enumerate(mask):
        for x, value in enumerate(row):
            if value:
                points.append((x, y))
    return points
