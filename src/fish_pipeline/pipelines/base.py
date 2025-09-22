"""Base pipeline abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Pipeline(ABC):
    """Simple abstraction used by the training and inference pipelines."""

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
