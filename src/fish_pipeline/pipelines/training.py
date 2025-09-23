"""Training pipeline for density models using pure Python data structures."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from pathlib import Path
from random import Random
from typing import Iterable, List, Sequence, Tuple

try:  # pragma: no cover - optional dependency guard
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    np = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency guard
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    Image = None  # type: ignore[assignment]

from ..models.density import BaseDensityModel, CSRNetDensityModel, LFCNetLikeDensityModel, TrainingSample
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
    dataset_root: str | None = None
    train_dir: str | None = None
    val_dir: str | None = None
    load_weights_path: str | None = None
    save_weights_dir: str | None = None
    save_weights_path: str | None = None
    best_checkpoint_name: str = "csrnet_best.pt"
    last_checkpoint_name: str = "csrnet_last.pt"


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

    def train(
        self,
        dataset: Sequence[TrainingSample],
        validation: Sequence[TrainingSample] | None = None,
        checkpoint_dir: Path | None = None,
        best_filename: str | None = None,
        last_filename: str | None = None,
    ) -> None:
        train_samples = list(dataset)
        val_samples = list(validation) if validation else None
        fit_method = getattr(self.model, "fit", None)
        if callable(fit_method):
            fit_kwargs = {
                "train_dataset": train_samples,
                "val_dataset": val_samples,
                "checkpoint_dir": checkpoint_dir,
                "best_filename": best_filename,
                "last_filename": last_filename,
            }
            # filter None values for models that may not accept them
            filtered_kwargs = {
                key: value
                for key, value in fit_kwargs.items()
                if value is not None or key in {"train_dataset", "val_dataset"}
            }
            fit_method(**filtered_kwargs)
            return
        self.model.train(train_samples)


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
        self.model = model or CSRNetDensityModel()
        self.config = config or TrainingConfig()

    def run(self) -> TrainingReport:
        if self.config.load_weights_path:
            self._load_weights(Path(self.config.load_weights_path))

        train_set, val_set = self._load_real_datasets()
        if not train_set:
            dataset = SyntheticDensityDataset(self.config).generate()
            split = int(len(dataset) * (1 - self.config.validation_split))
            train_set = dataset[:split]
            val_set = dataset[split:] if split < len(dataset) else dataset[-1:]

        checkpoint_dir = Path(self.config.save_weights_dir) if self.config.save_weights_dir else None
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer = DensityModelTrainer(self.model)
        trainer.train(
            train_set,
            validation=val_set,
            checkpoint_dir=checkpoint_dir,
            best_filename=self.config.best_checkpoint_name if checkpoint_dir else None,
            last_filename=self.config.last_checkpoint_name if checkpoint_dir else None,
        )

        final_save_path = Path(self.config.save_weights_path) if self.config.save_weights_path else None
        if final_save_path is not None:
            self._save_final_weights(final_save_path, checkpoint_dir)

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

    def _load_weights(self, path: Path) -> None:
        load_fn = getattr(self.model, "load_weights", None)
        if not callable(load_fn):
            return
        try:
            load_fn(path)
        except NotImplementedError:
            return

    def _save_final_weights(self, target_path: Path, checkpoint_dir: Path | None) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_dir is not None:
            best_candidate = checkpoint_dir / self.config.best_checkpoint_name
            if best_candidate.exists():
                target_path.write_bytes(best_candidate.read_bytes())
                return
        save_fn = getattr(self.model, "save_weights", None)
        if callable(save_fn):
            try:
                save_fn(target_path)
            except NotImplementedError:
                return

    def _load_real_datasets(self) -> Tuple[List[TrainingSample], List[TrainingSample]]:
        train_dir = self._resolve_split_path(self.config.train_dir, "train")
        val_dir = self._resolve_split_path(self.config.val_dir, "val")

        train_samples = self._load_split(train_dir)
        val_samples = self._load_split(val_dir)

        if train_samples and not val_samples:
            train_samples, val_samples = self._split_train_set(train_samples)

        return train_samples, val_samples

    def _resolve_split_path(self, override: str | None, default_subdir: str) -> Path | None:
        if override:
            return Path(override)
        if self.config.dataset_root:
            candidate = Path(self.config.dataset_root) / default_subdir
            if candidate.exists():
                return candidate
        return None

    def _load_split(self, split_dir: Path | None) -> List[TrainingSample]:
        if split_dir is None or not split_dir.exists():
            return []
        dataset = FileSystemDensityDataset(split_dir, random_seed=self.config.random_seed)
        return dataset.load()

    def _split_train_set(self, dataset: List[TrainingSample]) -> Tuple[List[TrainingSample], List[TrainingSample]]:
        if not dataset:
            return [], []
        split = int(len(dataset) * (1 - self.config.validation_split))
        if split <= 0:
            split = 1
        train_subset = dataset[:split]
        if not train_subset:
            train_subset = dataset[:1]
        if split < len(dataset):
            val_subset = dataset[split:]
        else:
            val_subset = dataset[-1:]
        return list(train_subset), list(val_subset)


class FileSystemDensityDataset:
    """Load density training samples from the documented folder layout."""

    SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    def __init__(self, split_dir: Path, random_seed: int) -> None:
        self.split_dir = split_dir
        self.random_seed = random_seed
        self.rgb_dir = split_dir / "rgb"
        self.density_dir = split_dir / "density"

    def load(self) -> List[TrainingSample]:
        if not self.rgb_dir.exists() or not self.density_dir.exists():
            return []
        samples: List[TrainingSample] = []
        rgb_files = {
            path.stem: path
            for path in self._iter_files(self.rgb_dir, self.SUPPORTED_IMAGE_EXTENSIONS)
        }
        density_extensions = set(self.SUPPORTED_IMAGE_EXTENSIONS)
        density_extensions.add(".npy")
        density_files = {
            path.stem: path
            for path in self._iter_files(self.density_dir, density_extensions)
        }
        paired_keys = sorted(set(rgb_files.keys()) & set(density_files.keys()))
        rng = Random(self.random_seed)
        rng.shuffle(paired_keys)
        for key in paired_keys:
            image_path = rgb_files[key]
            density_path = density_files[key]
            image = self._load_image_matrix(image_path)
            density = self._load_density_matrix(density_path)
            samples.append((image, density))
        return samples

    def _iter_files(self, directory: Path, allowed_extensions: set[str]) -> Iterable[Path]:
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in allowed_extensions:
                yield path

    def _load_image_matrix(self, path: Path) -> Matrix:
        if np is None or Image is None:
            raise RuntimeError(
                "Loading real datasets requires numpy and Pillow. Install them to enable disk datasets."
            )
        with Image.open(path) as img:
            grayscale = img.convert("F")
            array = np.asarray(grayscale, dtype=np.float32)
        if array.size == 0:
            return []
        max_val = float(array.max())
        if max_val > 0:
            array = array / max_val
        return array.tolist()

    def _load_density_matrix(self, path: Path) -> Matrix:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            if np is None:
                raise RuntimeError("Loading .npy density maps requires numpy.")
            array = np.load(path)
            array = np.asarray(array, dtype=np.float32)
            if array.ndim == 3:
                array = array.mean(axis=0)
            return array.tolist()
        if suffix in self.SUPPORTED_IMAGE_EXTENSIONS:
            if np is None or Image is None:
                raise RuntimeError(
                    "Loading image density maps requires numpy and Pillow."
                )
            with Image.open(path) as img:
                grayscale = img.convert("F")
                array = np.asarray(grayscale, dtype=np.float32)
            return array.tolist()
        raise ValueError(f"Unsupported density file format: {path.suffix}")
