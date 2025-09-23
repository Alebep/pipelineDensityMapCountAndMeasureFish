from datetime import datetime, timezone
from math import exp

from fish_pipeline.data.entities import (
    AlignmentRatios,
    CalibrationParameters,
    DepthInput,
    ImageInput,
    PipelineInput,
)
from fish_pipeline.pipelines.inference import (
    EndToEndMeasurementPipeline,
    InferenceConfig,
    InferenceRequest,
    SkeletonMeasurementPipeline,
)
from fish_pipeline.models.density import BaseDensityModel
from fish_pipeline.steps.segmentation import BasePromptedSegmenter, MaskInstance


def make_meta(width: int, height: int) -> PipelineInput:
    calibration = CalibrationParameters(
        fx=575.2,
        fy=575.0,
        cx=width / 2.0,
        cy=height / 2.0,
        alignment=AlignmentRatios(type="ratios", ratioW=1.0, ratioH=1.0),
    )
    image_meta = ImageInput(image_id="img_1", width=width, height=height)
    depth_meta = DepthInput(depth_id="depth_1", width_depth=width, height_depth=height, depth_units="mm")
    return PipelineInput(
        pid=123,
        timestamp_iso=datetime.now(timezone.utc).isoformat(),
        camera_name="realsense_d415",
        image=image_meta,
        depth=depth_meta,
        calibration=calibration,
    )


def make_blob_image(height: int, width: int, centers: list[tuple[int, int]]) -> list[list[float]]:
    image = [[0.0 for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            value = 0.0
            for cx, cy in centers:
                distance_sq = (x - cx) ** 2 + (y - cy) ** 2
                value += exp(-distance_sq / (2 * 25.0))
            image[y][x] = value
    return image


def make_depth_map(height: int, width: int, value: float = 1000.0) -> list[list[float]]:
    return [[value for _ in range(width)] for _ in range(height)]


def make_test_request(num_fish: int = 2) -> InferenceRequest:
    height = width = 96
    centers = [(30, 40), (65, 55)][:num_fish]
    image = make_blob_image(height, width, centers)
    depth_map = make_depth_map(height, width)
    meta = make_meta(width, height)
    return InferenceRequest(image=image, depth_map=depth_map, meta=meta)


def test_end_to_end_pipeline_output_structure():
    request = make_test_request()
    pipeline = EndToEndMeasurementPipeline(config=InferenceConfig())
    output = pipeline.run(request)

    assert output.counts.total_detected >= 2
    assert output.counts.total_measured == len(output.measurements)
    assert output.models.density_model == "CSRNet"
    assert all(measure.length.method == "end_to_end" for measure in output.measurements)

    result_dict = output.to_dict()
    assert "measurements" in result_dict
    assert result_dict["measurements"][0]["geometry"]["path"]["sampling"]["strategy"] == "arc_length"


def test_skeleton_pipeline_uses_fallback_when_needed():
    request = make_test_request()
    pipeline = SkeletonMeasurementPipeline(config=InferenceConfig())

    class FailingStrategy:
        method_name = "skeleton"

        def measure(self, *args, **kwargs):
            raise ValueError("failure")

    pipeline.measurement_strategy = FailingStrategy()
    output = pipeline.run(request)
    assert all(measure.length.method == "end_to_end" for measure in output.measurements)


def test_pipeline_accepts_custom_models():
    request = make_test_request()

    class DummyDensityModel(BaseDensityModel):
        def __init__(self) -> None:
            self.name = "DummyDensity"
            self.version = "test"

        def train(self, dataset):
            return None

        def predict(self, image):
            return image

    class DummySegmenter(BasePromptedSegmenter):
        model_name = "DummySegmenter"
        model_version = "1.0"

        def run(self, image, peaks):
            height = len(image)
            width = len(image[0]) if height else 0
            instances: list[MaskInstance] = []
            for peak in peaks:
                mask = [[False for _ in range(width)] for _ in range(height)]
                radius = 4
                for y in range(max(peak.y - radius, 0), min(peak.y + radius + 1, height)):
                    for x in range(max(peak.x - radius, 0), min(peak.x + radius + 1, width)):
                        if (x - peak.x) ** 2 + (y - peak.y) ** 2 <= radius ** 2:
                            mask[y][x] = True
                instances.append(
                    MaskInstance(mask=mask, center=(peak.x, peak.y), prompt_type="point", score=0.9)
                )
            return instances

    pipeline = EndToEndMeasurementPipeline(
        density_model=DummyDensityModel(),
        segmenter=DummySegmenter(),
        config=InferenceConfig(),
    )

    output = pipeline.run(request)

    assert output.models.density_model == "DummyDensity"
    assert output.models.sam_model == "DummySegmenter"
    assert output.counts.total_detected == len(output.measurements) >= 1
