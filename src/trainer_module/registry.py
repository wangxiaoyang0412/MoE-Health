"""Training-procedure (method) registry: one function per method trains and predicts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["METHOD_FACTORY", "MethodFactory", "MethodOutput", "register_method"]


@dataclass
class MethodOutput:
    """Validation / test predictions (dicts of arrays: prob, label, row, combo, ...)."""

    val: dict[str, np.ndarray]
    test: dict[str, np.ndarray]
    extra: dict[str, Any] = field(default_factory=dict)


METHOD_FACTORY: dict[str, Callable[..., MethodOutput]] = {}


def register_method(name: str) -> Callable[[Callable[..., MethodOutput]], Callable[..., MethodOutput]]:
    def decorator(fn: Callable[..., MethodOutput]) -> Callable[..., MethodOutput]:
        METHOD_FACTORY[name] = fn
        return fn

    return decorator


def MethodFactory(name: str) -> Callable[..., MethodOutput]:  # noqa: N802 - factory naming convention
    if name not in METHOD_FACTORY:
        raise KeyError(f"unknown method '{name}'; registered: {sorted(METHOD_FACTORY)}")
    return METHOD_FACTORY[name]
