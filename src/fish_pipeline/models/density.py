"""Density map model definitions, including a PyTorch CSRNet variant."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import exp
from random import Random
from typing import Any, Iterable, List, Sequence, Tuple

try:  # pragma: no cover - optional dependency guard
    import torch
    from torch import Tensor, nn
except ModuleNotFoundError:  # pragma: no cover - fallback when torch is unavailable
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[assignment]
    nn = Any  # type: ignore[assignment]

Matrix = List[List[float]]
TrainingSample = Tuple[Matrix, Matrix]


def _mean_filter(matrix: Matrix, kernel_size: int = 3) -> Matrix:
    if not matrix:
        return []
    height = len(matrix)
    width = len(matrix[0])
    radius = kernel_size // 2
    result = [[0.0 for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            values = []
            for ky in range(-radius, radius + 1):
                for kx in range(-radius, radius + 1):
                    ny = min(max(y + ky, 0), height - 1)
                    nx = min(max(x + kx, 0), width - 1)
                    values.append(matrix[ny][nx])
            result[y][x] = sum(values) / len(values)
    return result


def _normalize(matrix: Matrix) -> Matrix:
    if not matrix:
        return []
    flat = [value for row in matrix for value in row]
    min_val = min(flat)
    max_val = max(flat)
    if max_val == min_val:
        return [[0.0 for _ in row] for row in matrix]
    return [[(value - min_val) / (max_val - min_val) for value in row] for row in matrix]


def _matrix_sum(matrix: Matrix) -> float:
    return sum(sum(row) for row in matrix)


class BaseDensityModel(ABC):
    name: str = "BaseDensityModel"
    version: str = "0.0.1"

    @abstractmethod
    def train(self, dataset: Iterable[TrainingSample]) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: Matrix) -> Matrix:
        raise NotImplementedError


@dataclass
class LFCNetLikeDensityModel(BaseDensityModel):
    kernel_size: int = 3

    def __post_init__(self) -> None:
        self.name = "LFCNet"
        self.version = "1.2.0"
        self._scale = 1.0

    def train(self, dataset: Iterable[TrainingSample]) -> None:
        total_target = 0.0
        total_prediction = 0.0
        count = 0
        for image, target in dataset:
            prediction = self._feature_map(image)
            total_target += _matrix_sum(target)
            total_prediction += _matrix_sum(prediction)
            count += 1
        if count == 0:
            raise ValueError("Dataset cannot be empty")
        if total_prediction > 0:
            self._scale = total_target / total_prediction
        else:
            self._scale = 1.0

    def _feature_map(self, image: Matrix) -> Matrix:
        smoothed = _mean_filter(image, kernel_size=self.kernel_size)
        return _normalize(smoothed)

    def predict(self, image: Matrix) -> Matrix:
        feature = self._feature_map(image)
        return [[value * self._scale for value in row] for row in feature]

    def metadata(self) -> Tuple[str, str]:
        return self.name, self.version


if torch is not None:  # pragma: no cover - requires torch

    class _CSRNetBackbone(nn.Module):
        """A lightweight CSRNet-style convolutional backbone."""

        def __init__(self) -> None:
            super().__init__()
            self.frontend = nn.Sequential(
                self._conv_block(1, 32),
                self._conv_block(32, 64),
                self._conv_block(64, 128),
            )
            self.backend = nn.Sequential(
                self._dilated_block(128, 128, dilation=2),
                self._dilated_block(128, 64, dilation=2),
                self._dilated_block(64, 32, dilation=2),
            )
            self.output_layer = nn.Conv2d(32, 1, kernel_size=1)
            self._initialize_weights()

        @staticmethod
        def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        @staticmethod
        def _dilated_block(in_channels: int, out_channels: int, dilation: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                ),
                nn.ReLU(inplace=True),
            )

        def _initialize_weights(self) -> None:
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)

        def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
            x = self.frontend(inputs)
            x = self.backend(x)
            x = self.output_layer(x)
            return torch.relu(x)

else:  # pragma: no cover - fallback type for optional torch
    _CSRNetBackbone = None  # type: ignore[assignment]


class CSRNetDensityModel(BaseDensityModel):
    """PyTorch-based CSRNet model with lightweight warm-up training."""

    def __init__(
        self,
        lr: float = 1e-3,
        epochs: int = 3,
        batch_size: int = 4,
        device: str | None = None,
        warmup_samples: int = 6,
        warmup_epochs: int = 2,
    ) -> None:
        self.name = "CSRNet"
        self.version = "pytorch-0.1"
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self._scale = 1.0
        self._last_loss: float | None = None
        self._is_trained = False
        self._fallback_model: LFCNetLikeDensityModel | None = None

        if torch is None or _CSRNetBackbone is None:
            self.device = "cpu"
            self.network = None
            self.criterion = None
            self.optimizer = None
            fallback = LFCNetLikeDensityModel(kernel_size=5)
            fallback.name = self.name
            fallback.version = "fallback"
            self._fallback_model = fallback
            self._is_trained = True
            return

        torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = torch_device
        self.network = _CSRNetBackbone().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        if warmup_samples > 0 and warmup_epochs > 0:
            self._warmup(warmup_samples, warmup_epochs)

    def train(self, dataset: Iterable[TrainingSample]) -> None:
        samples = list(dataset)
        if not samples:
            raise ValueError("Dataset cannot be empty")
        if self._fallback_model is not None:
            self._fallback_model.train(samples)
            self._scale = getattr(self._fallback_model, "_scale", 1.0)
            return
        inputs, targets = self._prepare_tensors(samples)
        self._run_optimization(inputs, targets, epochs=self.epochs)
        self._calibrate_scale(inputs, targets)
        self._is_trained = True

    def predict(self, image: Matrix) -> Matrix:
        if not image:
            return []
        if self._fallback_model is not None:
            return self._fallback_model.predict(image)
        if not self._is_trained:
            return _mean_filter(image, kernel_size=5)
        tensor = torch.tensor(image, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        self.network.eval()
        with torch.no_grad():
            prediction = self.network(tensor) * self._scale
        result = prediction.squeeze(0).squeeze(0).cpu().tolist()
        return result

    def metadata(self) -> Tuple[str, str]:
        return self.name, self.version

    def _prepare_tensors(self, samples: Sequence[TrainingSample]) -> Tuple[Tensor, Tensor]:
        if torch is None:
            raise RuntimeError("PyTorch is required for CSRNet training")
        images = [torch.tensor(image, dtype=torch.float32) for image, _ in samples]
        targets = [torch.tensor(target, dtype=torch.float32) for _, target in samples]
        inputs = torch.stack(images).unsqueeze(1).to(self.device)
        labels = torch.stack(targets).unsqueeze(1).to(self.device)
        return inputs, labels

    def _run_optimization(self, inputs: Tensor, targets: Tensor, epochs: int) -> None:
        if torch is None:
            return
        if inputs.shape[0] == 0:
            return
        self.network.train()
        for _ in range(epochs):
            permutation = torch.randperm(inputs.shape[0], device=self.device)
            inputs_epoch = inputs[permutation]
            targets_epoch = targets[permutation]
            for start in range(0, inputs_epoch.shape[0], self.batch_size):
                end = start + self.batch_size
                batch_x = inputs_epoch[start:end]
                batch_y = targets_epoch[start:end]
                self.optimizer.zero_grad()
                output = self.network(batch_x)
                loss = self.criterion(output, batch_y)
                loss.backward()
                self.optimizer.step()
                self._last_loss = float(loss.detach().cpu().item())

    def _calibrate_scale(self, inputs: Tensor, targets: Tensor) -> None:
        if torch is None:
            return
        with torch.no_grad():
            preds = self.network(inputs)
        pred_sum = float(preds.sum().cpu().item())
        target_sum = float(targets.sum().cpu().item())
        if pred_sum > 0:
            self._scale = target_sum / pred_sum
        else:
            self._scale = 1.0

    def _warmup(self, num_samples: int, epochs: int) -> None:
        if torch is None:
            return
        rng = Random(1234)
        samples: list[TrainingSample] = []
        height = width = 64
        for _ in range(num_samples):
            cx = rng.uniform(8, width - 8)
            cy = rng.uniform(8, height - 8)
            sigma = rng.uniform(3.0, 6.0)
            image = self._gaussian_blob(height, width, cx, cy, sigma)
            density = _normalize(image)
            samples.append((density, density))
        inputs, targets = self._prepare_tensors(samples)
        self._run_optimization(inputs, targets, epochs=epochs)
        self._calibrate_scale(inputs, targets)
        self._is_trained = True

    def _gaussian_blob(self, height: int, width: int, cx: float, cy: float, sigma: float) -> Matrix:
        matrix: Matrix = []
        for y in range(height):
            row: list[float] = []
            for x in range(width):
                distance_sq = (x - cx) ** 2 + (y - cy) ** 2
                value = exp(-distance_sq / (2.0 * sigma ** 2))
                row.append(value)
            matrix.append(row)
        return matrix
