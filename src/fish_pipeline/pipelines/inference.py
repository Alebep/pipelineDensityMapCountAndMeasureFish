"""Inference pipelines for fish length measurement."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from ..data.entities import (
    CountsSummary,
    InferenceOutput,
    LengthMeasurement,
    MeasurementGeometry,
    MeasurementRecord,
    ModelsMetadata,
    PathInfo,
    PathSampling,
    PipelineInput,
    ProcessingMetadata,
)
from ..models.density import BaseDensityModel, LFCNetLikeDensityModel
from ..steps.density import DensityMapGenerator
from ..steps.measurement import (
    EndToEndMeasurementStrategy,
    LengthMeasurementStrategy,
    MeasurementComputation,
    SkeletonMeasurementStrategy,
)
from ..steps.peaks import PeakDetector
from ..steps.preprocessing import Preprocessor, PreprocessingConfig
from ..steps.qc import QualityController
from ..steps.segmentation import MaskInstance, PromptedSegmenter
from ..utils.calibration import DepthProjector
from .base import Pipeline


@dataclass
class InferenceConfig:
    preproc: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    host: str = "worker-01"
    pipeline_version: str = "rgbd-0.9.3"
    min_area: int = 150
    sampling_step_px: float = 4.0


@dataclass
class InferenceRequest:
    image: List[List[float]]
    depth_map: List[List[float]]
    meta: PipelineInput


class BaseInferencePipeline(Pipeline):
    def __init__(
        self,
        density_model: Optional[BaseDensityModel] = None,
        measurement_strategy: Optional[LengthMeasurementStrategy] = None,
        fallback_strategy: Optional[LengthMeasurementStrategy] = None,
        config: Optional[InferenceConfig] = None,
    ) -> None:
        self.density_model = density_model or LFCNetLikeDensityModel()
        self.segmenter = PromptedSegmenter()
        self.peak_detector = PeakDetector()
        self.measurement_strategy = measurement_strategy or EndToEndMeasurementStrategy()
        self.fallback_strategy = fallback_strategy
        self.config = config or InferenceConfig()
        self.preprocessor = Preprocessor(self.config.preproc)
        self.qc = QualityController(min_area=self.config.min_area)
        self.density_step = DensityMapGenerator(self.density_model)

    def run(self, request: InferenceRequest) -> InferenceOutput:
        start_time = time.time()
        image_proc, depth_proc = self.preprocessor.run(request.image, request.depth_map)
        density_result = self.density_step.run(image_proc)
        peaks = self.peak_detector.run(density_result.density_map)
        masks = self.segmenter.run(image_proc, peaks)

        projector = DepthProjector(request.meta.calibration)
        measurements: List[MeasurementRecord] = []

        for instance in masks:
            measurement = self._measure_instance(
                instance=instance,
                depth_map=depth_proc,
                depth_units=request.meta.depth.depth_units,
                projector=projector,
            )
            if measurement:
                measurements.append(measurement)

        models_metadata = ModelsMetadata(
            density_model=self.density_model.name,
            density_model_version=self.density_model.version,
            sam_model=self.segmenter.model_name,
            sam_model_version=self.segmenter.model_version,
        )

        counts = CountsSummary(
            total_detected=len(masks),
            total_measured=len(measurements),
        )

        runtime_ms = int((time.time() - start_time) * 1000)

        processing = ProcessingMetadata(
            runtime_ms=runtime_ms,
            host=self.config.host,
            pipeline_version=self.config.pipeline_version,
        )

        meta = request.meta
        output = InferenceOutput(
            pid=meta.pid,
            timestamp_iso=meta.timestamp_iso,
            camera_name=meta.camera_name,
            width=meta.image.width,
            height=meta.image.height,
            width_depth=meta.depth.width_depth,
            height_depth=meta.depth.height_depth,
            depth_units=meta.depth.depth_units,
            models=models_metadata,
            calibration=meta.calibration,
            counts=counts,
            measurements=measurements,
            processing=processing,
        )
        return output

    def _measure_instance(
        self,
        instance: MaskInstance,
        depth_map: List[List[float]],
        depth_units: str,
        projector: DepthProjector,
    ) -> Optional[MeasurementRecord]:
        strategies = [self.measurement_strategy]
        if self.fallback_strategy:
            strategies.append(self.fallback_strategy)

        for strategy in strategies:
            try:
                computation = strategy.measure(
                    mask=instance.mask,
                    head_point=(instance.center[0], instance.center[1]),
                    depth_map=depth_map,
                    depth_units=depth_units,
                    projector=projector,
                )
                return self._build_measurement_record(instance, computation)
            except ValueError:
                continue
        return None

    def _build_measurement_record(
        self,
        instance: MaskInstance,
        computation: MeasurementComputation,
    ) -> MeasurementRecord:
        path_points = [[float(x), float(y)] for x, y in computation.path_rgb]
        path_type = "segment" if len(path_points) == 2 else "polyline"
        sampling = PathSampling(strategy="arc_length", pixel_step=self.config.sampling_step_px)

        endpoints = {
            "rgb_px": {
                "head": {"x": computation.head[0], "y": computation.head[1]},
                "tail": {"x": computation.tail[0], "y": computation.tail[1]},
            }
        }

        endpoints_3d = self._extract_3d_endpoints(computation.depth_points)

        depth_valid_fraction = computation.depth_valid_fraction
        raw_confidence = (instance.score + depth_valid_fraction) / 2.0
        confidence = max(0.0, min(raw_confidence, 1.0))

        qc_eval = self.qc.evaluate(
            mask=instance.mask,
            prompt_type=instance.prompt_type,
            depth_valid_fraction=depth_valid_fraction,
            method_used=computation.method,
            confidence=confidence,
        )

        geometry = MeasurementGeometry(
            endpoints=endpoints,
            endpoints_3d=endpoints_3d,
            path_info=PathInfo(type=path_type, num_points=len(path_points)),
            path={
                "rgb_px": path_points,
                "sampling": sampling,
            },
        )

        length_info = LengthMeasurement(
            px=computation.length_px,
            mm=computation.length_mm,
            method=computation.method,
            segments_used=computation.segments_used,
        )
        record = MeasurementRecord(
            center_px={"x": float(instance.center[0]), "y": float(instance.center[1])},
            prompt_type=instance.prompt_type,
            mask_stats=qc_eval.mask_stats,
            qc=qc_eval.qc,
            length=length_info,
            geometry=geometry,
            errors=qc_eval.errors,
        )
        return record

    def _extract_3d_endpoints(
        self, depth_points: List[Tuple[float, float, float]]
    ) -> dict[str, dict[str, dict[str, float]]]:
        result: dict[str, dict[str, dict[str, float]]] = {}
        valid_indices = [idx for idx, point in enumerate(depth_points) if not self._has_nan(point)]
        if not valid_indices:
            return result

        first_idx = valid_indices[0]
        last_idx = valid_indices[-1]
        head_point = depth_points[first_idx]
        tail_point = depth_points[last_idx]

        if first_idx == 0 and last_idx == len(depth_points) - 1 and first_idx != last_idx:
            result["head_tail_mm"] = {
                "head": self._point_to_dict(head_point),
                "tail": self._point_to_dict(tail_point),
            }
        else:
            if first_idx < len(depth_points) - 1:
                next_point = depth_points[min(first_idx + 1, len(depth_points) - 1)]
                result["first_valid_segment_mm"] = {
                    "head": self._point_to_dict(head_point),
                    "tail": self._point_to_dict(next_point),
                }
            if last_idx > 0:
                prev_point = depth_points[max(last_idx - 1, 0)]
                result["last_valid_segment_mm"] = {
                    "p1": self._point_to_dict(prev_point),
                    "p2": self._point_to_dict(tail_point),
                }
        return result

    def _point_to_dict(self, point: Tuple[float, float, float]) -> dict[str, float]:
        return {"x": float(point[0]), "y": float(point[1]), "z": float(point[2])}

    def _has_nan(self, point: Tuple[float, float, float]) -> bool:
        return any(value != value for value in point)


class EndToEndMeasurementPipeline(BaseInferencePipeline):
    def __init__(self, config: Optional[InferenceConfig] = None) -> None:
        super().__init__(
            measurement_strategy=EndToEndMeasurementStrategy(),
            config=config,
        )


class SkeletonMeasurementPipeline(BaseInferencePipeline):
    def __init__(self, config: Optional[InferenceConfig] = None) -> None:
        super().__init__(
            measurement_strategy=SkeletonMeasurementStrategy(),
            fallback_strategy=EndToEndMeasurementStrategy(),
            config=config,
        )
