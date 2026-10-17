"""Dataset factory and registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["DATASET_FACTORY", "DatasetFactory", "register_dataset"]

DATASET_FACTORY: dict[str, Callable[..., Any]] = {}


def register_dataset(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a dataset builder under ``name``."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        DATASET_FACTORY[name] = fn
        return fn

    return decorator


def DatasetFactory(name: str) -> Callable[..., Any]:  # noqa: N802 - factory naming convention
    if name not in DATASET_FACTORY:
        raise KeyError(f"unknown dataset '{name}'; registered: {sorted(DATASET_FACTORY)}")
    return DATASET_FACTORY[name]
