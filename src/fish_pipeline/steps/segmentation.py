"""Instance segmentation powered by SAM-HQ."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Iterable, List, Tuple

try:  # pragma: no cover - optional dependency guard
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    np = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency guard
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    torch = None  # type: ignore[assignment]

if torch is not None:  # pragma: no cover - optional dependency guard
    try:
        from segment_anything_hq import SamPredictor, sam_model_registry
    except ModuleNotFoundError:  # pragma: no cover - dependency missing
        SamPredictor = None  # type: ignore[assignment]
        sam_model_registry = {}
    except Exception:  # pragma: no cover - other import failures
        SamPredictor = None  # type: ignore[assignment]
        sam_model_registry = {}
else:  # pragma: no cover - torch not available
    SamPredictor = None  # type: ignore[assignment]
    sam_model_registry = {}

from .peaks import Peak

Mask = List[List[bool]]


def _package_version() -> str:
    try:  # pragma: no cover - optional dependency metadata
        return metadata.version("segment-anything-hq")
    except metadata.PackageNotFoundError:  # type: ignore[attr-defined]
        return "unavailable"
    except Exception:
        return "unknown"


@dataclass
class MaskInstance:
    mask: Mask
    center: Tuple[int, int]
    prompt_type: str
    score: float


class BasePromptedSegmenter(ABC):
    """Abstract base class for promptable segmenters."""

    model_name: str = "prompted-segmenter"
    model_version: str = "0.0.0"

    @abstractmethod
    def run(self, image: List[List[float]], peaks: Iterable[Peak]) -> List[MaskInstance]:
        """Produce segmentation masks for each peak."""

    def metadata(self) -> Tuple[str, str]:
        return self.model_name, self.model_version


class SAMHQSegmenter(BasePromptedSegmenter):
    """Wrapper around the official SAM-HQ predictor."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        model_type: str = "vit_b",
        device: str | None = None,
        multimask_output: bool = False,
        mask_threshold: float = 0.5,
        use_hq_token_only: bool = True,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.model_type = model_type
        self.requested_device = device
        self.multimask_output = multimask_output
        self.mask_threshold = mask_threshold
        self.use_hq_token_only = use_hq_token_only
        self.model_name = "SAM-HQ"
        self.model_version = _package_version()
        self._predictor: SamPredictor | None = None

    def run(self, image: List[List[float]], peaks: Iterable[Peak]) -> List[MaskInstance]:  # type: ignore[override]
        if not image:
            return []
        predictor = self._get_predictor()
        if predictor is None:
            return self._fallback_segmentation(image, peaks)
        np_image = self._prepare_image(image)
        predictor.set_image(np_image, image_format="RGB")

        instances: List[MaskInstance] = []
        for peak in peaks:
            mask_bool, score = self._predict_single(predictor, peak)
            instances.append(
                MaskInstance(
                    mask=mask_bool,
                    center=(peak.x, peak.y),
                    prompt_type="point",
                    score=score,
                )
            )
        return instances

    def _get_predictor(self) -> SamPredictor | None:
        if self._predictor is not None:
            return self._predictor
        if np is None or torch is None:
            self.model_version = "fallback"
            return None
        if SamPredictor is None or not sam_model_registry:
            self.model_version = "fallback"
            return None

        builder = sam_model_registry.get(self.model_type) or sam_model_registry.get("default")
        if builder is None:
            raise ValueError(f"Unknown SAM-HQ model type: {self.model_type}")

        checkpoint = self._resolve_checkpoint()
        sam_model = builder(checkpoint=str(checkpoint) if checkpoint else None)
        device_str = self.requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
        sam_model.to(device_str)
        self._predictor = SamPredictor(sam_model)
        return self._predictor

    def _resolve_checkpoint(self) -> Path | None:
        if self.checkpoint_path is None:
            return None
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"SAM-HQ checkpoint not found: {self.checkpoint_path}")
        return self.checkpoint_path

    def _prepare_image(self, image: List[List[float]]) -> "np.ndarray":
        if np is None:
            raise RuntimeError("NumPy is required to prepare images for SAM-HQ.")
        array = np.asarray(image, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("SAM-HQ expects single-channel grayscale inputs.")
        if array.size == 0:
            raise ValueError("Input image is empty.")
        normalized = np.clip(array, 0.0, 1.0)
        scaled = (normalized * 255.0).astype(np.uint8)
        rgb = np.stack([scaled, scaled, scaled], axis=-1)
        return rgb

    def _predict_single(self, predictor: SamPredictor, peak: Peak) -> Tuple[Mask, float]:
        if np is None:
            raise RuntimeError("NumPy is required to run SAM-HQ.")
        point_coords = np.array([[float(peak.x), float(peak.y)]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)
        masks, iou_predictions, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=self.multimask_output,
            return_logits=False,
            hq_token_only=self.use_hq_token_only,
        )
        if masks.ndim != 3 or iou_predictions.ndim != 1:
            raise RuntimeError("Unexpected SAM-HQ output shape.")
        best_index = int(iou_predictions.argmax())
        mask = masks[best_index] >= self.mask_threshold
        mask_bool: Mask = mask.astype(bool).tolist()
        score = float(iou_predictions[best_index])
        return mask_bool, score

    def _fallback_segmentation(self, image: List[List[float]], peaks: Iterable[Peak]) -> List[MaskInstance]:
        height = len(image)
        width = len(image[0]) if height else 0
        if height == 0 or width == 0:
            return []
        # Simple circular masks around peaks as a graceful degradation path.
        radius = max(4, min(height, width) // 12)
        instances: List[MaskInstance] = []
        for peak in peaks:
            mask = [[False for _ in range(width)] for _ in range(height)]
            for y in range(max(peak.y - radius, 0), min(peak.y + radius + 1, height)):
                for x in range(max(peak.x - radius, 0), min(peak.x + radius + 1, width)):
                    if (x - peak.x) ** 2 + (y - peak.y) ** 2 <= radius ** 2:
                        mask[y][x] = True
            instances.append(
                MaskInstance(
                    mask=mask,
                    center=(peak.x, peak.y),
                    prompt_type="point",
                    score=0.0,
                )
            )
        return instances


PromptedSegmenter = SAMHQSegmenter

