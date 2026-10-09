"""Training utilities for PriceFlow."""

from importlib import import_module
from typing import Any

__all__ = ["load_trained_model", "main", "run_training"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return getattr(import_module(".train_agent", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
