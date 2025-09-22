"""Data entities used across the fish measurement pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from ..utils.geometry import (
    OrientedBBox,
    ensure_clockwise_quadrilateral,
    oriented_bbox_with_ratio,
    polygon_area,
)


class MeasureMode(Enum):
    """Enumeration of supported measurement modes."""

    KEYPOINT = 1
    OUTLINE = 2


@dataclass
class Measure:
    """Representation of a single measurement, created by humans or AI."""

    mid: int
    type: int
    keypoints: Optional[Sequence[float]] = None
    score: float = 100.0
    fishid: Optional[int] = None
    angle: Optional[float] = None
    measure_mode: MeasureMode = MeasureMode.KEYPOINT
    distance_mm: Optional[float] = None
    user_id: Optional[str] = None

    def __post_init__(self) -> None:
        self.kps: List[float] = []
        self.ray_kps_aabb: List[float] = []
        self.start_x: Optional[float] = None
        self.start_y: Optional[float] = None
        self.finish_x: Optional[float] = None
        self.finish_y: Optional[float] = None

        if self.keypoints is not None and len(self.keypoints) % 2 == 0 and len(self.keypoints) >= 4:
            self.kps = list(self.keypoints)
            self.start_x = self.kps[0]
            self.start_y = self.kps[1]
            self.finish_x = self.kps[-2]
            self.finish_y = self.kps[-1]

    @property
    def mode(self) -> MeasureMode:
        return self.measure_mode

    @property
    def measure_id(self) -> int:
        return self.mid

    @property
    def type_m(self) -> int:
        return self.type

    @property
    def dist_mm(self) -> Optional[float]:
        return self.distance_mm

    @dist_mm.setter
    def dist_mm(self, value: Optional[float]) -> None:
        self.distance_mm = value

    @property
    def conf(self) -> float:
        return self.score

    @property
    def user(self) -> Optional[str]:
        return self.user_id

    def obb(self, ratio: float = 3.0) -> Tuple[OrientedBBox, List[Tuple[float, float]]]:
        """Return an oriented bounding box with a fixed aspect ratio."""

        if not self.kps:
            raise ValueError("No keypoints available to compute OBB")

        points = [(self.kps[i], self.kps[i + 1]) for i in range(0, len(self.kps), 2)]
        return oriented_bbox_with_ratio(points, ratio=ratio)

    def aabb_shifted(
        self, img_h: int, img_w: int, ratio: float = 3.0, quadrilateral: bool = True
    ) -> Tuple[int, int, int, int] | List[Tuple[float, float]]:
        """Generate an axis aligned bounding box derived from the oriented box."""

        _, obb_pts = self.obb(ratio)
        xs = [pt[0] for pt in obb_pts]
        ys = [pt[1] for pt in obb_pts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        w = x_max - x_min
        h = y_max - y_min

        if w > img_w:
            x_min = 0
            x_max = img_w - 1
        else:
            if x_min < 0:
                x_min, x_max = 0, w
            elif x_max >= img_w:
                x_min, x_max = img_w - 1 - w, img_w - 1

        if h > img_h:
            y_min = 1
            y_max = img_h - 2
        else:
            if y_min < 0:
                y_min, y_max = 1, h + 1
            elif y_max >= img_h:
                y_min, y_max = img_h - 2 - h, img_h - 2
            else:
                y_min = max(1, y_min)
                y_max = min(img_h - 2, y_max)

        x_min = int(round(max(0, min(x_min, img_w - 1))))
        y_min = int(round(max(0, min(y_min, img_h - 1))))
        x_max = int(round(max(0, min(x_max, img_w - 1))))
        y_max = int(round(max(0, min(y_max, img_h - 1))))

        if quadrilateral:
            return ensure_clockwise_quadrilateral(x_min, y_min, x_max, y_max)
        return x_min, y_min, x_max, y_max


@dataclass
class Sample:
    pid: int
    specie_id: int
    seaport_id: int
    state_id: int
    size: int
    capture_time: datetime
    measures: List[Measure]

    def pretty(self, indent: int = 0) -> str:
        pad = " " * indent
        info = [
            (
                f"{pad}Sample(pid={self.pid}, specie_id={self.specie_id}, "
                f"seaport_id={self.seaport_id}, state_id={self.state_id}, "
                f"size={self.size}, capture_time={self.capture_time})"
            )
        ]

        if not self.measures:
            info.append(f"{pad}  └─ (sem medidas)")
        else:
            for idx, measure in enumerate(self.measures, 1):
                kp_str = " ".join(map(str, measure.kps)) if measure.kps else "[]"
                info.append(
                    f"{pad}  ├─ Measure #{idx}: id={measure.measure_id}, "
                    f"type={measure.type_m}, dist_mm={measure.dist_mm}, "
                    f"mode={measure.mode.name}, user={measure.user}, "
                    f"fishid={measure.fishid}, kps=[{kp_str}]"
                )
        return "\n".join(info)

    def print(self) -> None:
        print(self.pretty())


# === Pipeline data transfer objects ===


def now_iso8601() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AlignmentRatios:
    type: str
    ratioW: float
    ratioH: float

    def to_dict(self) -> Dict[str, float | str]:
        return asdict(self)


@dataclass
class CalibrationParameters:
    fx: float
    fy: float
    cx: float
    cy: float
    alignment: AlignmentRatios

    def to_dict(self) -> Dict[str, float | Dict[str, float | str]]:
        result = asdict(self)
        result["alignment"] = self.alignment.to_dict()
        return result


@dataclass
class ImageInput:
    image_id: str
    width: int
    height: int


@dataclass
class DepthInput:
    depth_id: str
    width_depth: int
    height_depth: int
    depth_units: str


@dataclass
class PipelineInput:
    pid: int
    timestamp_iso: str
    camera_name: str
    image: ImageInput
    depth: DepthInput
    calibration: CalibrationParameters


@dataclass
class MaskStats:
    area_px: int
    bbox: Dict[str, int]
    touches_border: bool
    iou_suppressed: bool


@dataclass
class QualityControl:
    mask_ok: bool
    depth_valid_fraction: float
    occlusion_flag: bool
    method_used: str
    confidence: float


@dataclass
class LengthMeasurement:
    px: float
    mm: float
    method: str
    segments_used: int


@dataclass
class PathInfo:
    type: str
    num_points: int


@dataclass
class PathSampling:
    strategy: str
    pixel_step: float


@dataclass
class MeasurementGeometry:
    endpoints: Dict[str, Dict[str, Dict[str, float]]]
    endpoints_3d: Dict[str, Dict[str, Dict[str, float]]]
    path_info: PathInfo
    path: Dict[str, List[List[float]] | PathSampling]


@dataclass
class MeasurementError:
    messages: List[str] = field(default_factory=list)


@dataclass
class MeasurementRecord:
    center_px: Dict[str, float]
    prompt_type: str
    mask_stats: MaskStats
    qc: QualityControl
    length: LengthMeasurement
    geometry: MeasurementGeometry
    errors: List[str]
    measure_id: Optional[int] = None
    fish_id: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:
        result = asdict(self)
        result["mask_stats"] = asdict(self.mask_stats)
        result["qc"] = asdict(self.qc)
        result["length"] = asdict(self.length)
        result["geometry"] = {
            "endpoints": self.geometry.endpoints,
            "endpoints_3d": self.geometry.endpoints_3d,
            "path_info": asdict(self.geometry.path_info),
            "path": {
                key: asdict(value) if is_dataclass(value) else value
                for key, value in self.geometry.path.items()
            },
        }
        return result


@dataclass
class ModelsMetadata:
    density_model: str
    density_model_version: str
    sam_model: str
    sam_model_version: str


@dataclass
class CountsSummary:
    total_detected: int
    total_measured: int


@dataclass
class ProcessingMetadata:
    runtime_ms: int
    host: str
    pipeline_version: str


@dataclass
class InferenceOutput:
    pid: int
    timestamp_iso: str
    camera_name: str
    width: int
    height: int
    width_depth: int
    height_depth: int
    depth_units: str
    models: ModelsMetadata
    calibration: CalibrationParameters
    counts: CountsSummary
    measurements: List[MeasurementRecord]
    processing: ProcessingMetadata

    def to_dict(self) -> Dict[str, object]:
        return {
            "pid": self.pid,
            "timestamp_iso": self.timestamp_iso,
            "camera_name": self.camera_name,
            "width": self.width,
            "height": self.height,
            "width_depth": self.width_depth,
            "height_depth": self.height_depth,
            "depth_units": self.depth_units,
            "models": asdict(self.models),
            "calibration": self.calibration.to_dict(),
            "counts": asdict(self.counts),
            "measurements": [record.to_dict() for record in self.measurements],
            "processing": asdict(self.processing),
        }


@dataclass
class Polygon:
    points: List[Tuple[float, float]]

    @property
    def area(self) -> float:
        return polygon_area(self.points)


__all__ = [
    "MeasureMode",
    "Measure",
    "Sample",
    "AlignmentRatios",
    "CalibrationParameters",
    "ImageInput",
    "DepthInput",
    "PipelineInput",
    "MaskStats",
    "QualityControl",
    "LengthMeasurement",
    "MeasurementGeometry",
    "MeasurementRecord",
    "ModelsMetadata",
    "CountsSummary",
    "ProcessingMetadata",
    "InferenceOutput",
    "Polygon",
]
