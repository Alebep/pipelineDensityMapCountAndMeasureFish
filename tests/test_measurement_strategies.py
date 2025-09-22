from fish_pipeline.data.entities import AlignmentRatios, CalibrationParameters
from fish_pipeline.steps.measurement import (
    EndToEndMeasurementStrategy,
    SkeletonMeasurementStrategy,
)
from fish_pipeline.utils.calibration import DepthProjector


def build_projector(width: int, height: int) -> DepthProjector:
    calibration = CalibrationParameters(
        fx=500.0,
        fy=500.0,
        cx=width / 2.0,
        cy=height / 2.0,
        alignment=AlignmentRatios(type="ratios", ratioW=1.0, ratioH=1.0),
    )
    return DepthProjector(calibration)


def build_mask(height: int, width: int) -> list[list[bool]]:
    mask = [[False for _ in range(width)] for _ in range(height)]
    for y in range(18, 22):
        for x in range(5, 30):
            mask[y][x] = True
    for y in range(22, 32):
        for x in range(27, 31):
            mask[y][x] = True
    return mask


def build_depth_map(height: int, width: int, value: float = 1000.0) -> list[list[float]]:
    return [[value for _ in range(width)] for _ in range(height)]


def test_skeleton_longer_than_end_to_end_for_curved_mask():
    height, width = 40, 40
    mask = build_mask(height, width)
    depth_map = build_depth_map(height, width)
    projector = build_projector(width, height)

    head_point = (7, 20)

    end_to_end = EndToEndMeasurementStrategy()
    skeleton = SkeletonMeasurementStrategy()

    end_result = end_to_end.measure(mask, head_point, depth_map, "mm", projector)
    skeleton_result = skeleton.measure(mask, head_point, depth_map, "mm", projector)

    assert skeleton_result.length_px > end_result.length_px
    assert skeleton_result.method == "skeleton"
    assert end_result.method == "end_to_end"
    assert len(skeleton_result.path_rgb) > 2
