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
    assert output.models.density_model == "LFCNet"
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
