"""Training pipeline for density models using pure Python data structures."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from random import Random
from typing import List, Sequence, Tuple

from ..models.density import BaseDensityModel, LFCNetLikeDensityModel, TrainingSample
from .base import Pipeline

Matrix = List[List[float]]


def _matrix_sum(matrix: Matrix) -> float:
    return sum(sum(row) for row in matrix)


def _normalize(matrix: Matrix) -> Matrix:
    flat = [value for row in matrix for value in row]
    min_val = min(flat)
    max_val = max(flat)
    if max_val == min_val:
        return [[0.0 for _ in row] for row in matrix]
    return [[(value - min_val) / (max_val - min_val) for value in row] for row in matrix]


@dataclass
class TrainingConfig:
    num_samples: int = 20
    image_size: Tuple[int, int] = (64, 64)
    validation_split: float = 0.2
    random_seed: int = 42


@dataclass
class TrainingReport:
    training_samples: int
    validation_samples: int
    count_mae: float
    count_rmse: float
    scale: float


class SyntheticDensityDataset:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.rng = Random(config.random_seed)

    def generate(self) -> List[TrainingSample]:
        samples: List[TrainingSample] = []
        height, width = self.config.image_size
        for _ in range(self.config.num_samples):
            image = self._generate_image(height, width)
            density = _normalize(image)
            samples.append((image, density))
        return samples

    def _generate_image(self, height: int, width: int) -> Matrix:
        image = [[0.0 for _ in range(width)] for _ in range(height)]
        num_fish = self.rng.randint(1, 3)
        centers = []
        for _ in range(num_fish):
            cx = self.rng.uniform(5, width - 5)
            cy = self.rng.uniform(5, height - 5)
            radius = self.rng.uniform(2.0, 4.0)
            centers.append((cx, cy, radius))
        for y in range(height):
            for x in range(width):
                value = 0.0
                for cx, cy, radius in centers:
                    distance_sq = (x - cx) ** 2 + (y - cy) ** 2
                    value += exp(-distance_sq / (2 * radius ** 2))
                image[y][x] = value
        return _normalize(image)


class DensityModelTrainer:
    def __init__(self, model: BaseDensityModel) -> None:
        self.model = model

    def train(self, dataset: Sequence[TrainingSample]) -> None:
        self.model.train(dataset)


class DensityModelEvaluator:
    def __init__(self, model: BaseDensityModel) -> None:
        self.model = model

    def evaluate(self, dataset: Sequence[TrainingSample]) -> Tuple[float, float]:
        errors = []
        squared_errors = []
        for image, target in dataset:
            prediction = self.model.predict(image)
            errors.append(abs(_matrix_sum(prediction) - _matrix_sum(target)))
            diff = _matrix_sum(prediction) - _matrix_sum(target)
            squared_errors.append(diff * diff)
        mae = sum(errors) / len(errors) if errors else 0.0
        rmse = (sum(squared_errors) / len(squared_errors)) ** 0.5 if squared_errors else 0.0
        return mae, rmse


class DensityModelTrainingPipeline(Pipeline):
    def __init__(
        self,
        model: BaseDensityModel | None = None,
        config: TrainingConfig | None = None,
    ) -> None:
        self.model = model or LFCNetLikeDensityModel()
        self.config = config or TrainingConfig()

    def run(self) -> TrainingReport:
        dataset = SyntheticDensityDataset(self.config).generate()
        split = int(len(dataset) * (1 - self.config.validation_split))
        train_set = dataset[:split]
        val_set = dataset[split:] if split < len(dataset) else dataset[-1:]

        trainer = DensityModelTrainer(self.model)
        trainer.train(train_set)

        evaluator = DensityModelEvaluator(self.model)
        mae, rmse = evaluator.evaluate(val_set)

        scale = getattr(self.model, "_scale", 1.0)
        return TrainingReport(
            training_samples=len(train_set),
            validation_samples=len(val_set),
            count_mae=mae,
            count_rmse=rmse,
            scale=float(scale),
        )
