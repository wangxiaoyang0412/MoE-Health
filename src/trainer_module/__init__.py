"""Trainer module: training loop and one registered procedure per method."""

from src.trainer_module import moe_procedure, unimodal_procedure  # noqa: F401
from src.trainer_module.registry import METHOD_FACTORY, MethodFactory, MethodOutput, register_method
from src.trainer_module.trainer import FitResult, fit, predict

__all__ = [
    "METHOD_FACTORY",
    "FitResult",
    "MethodFactory",
    "MethodOutput",
    "fit",
    "predict",
    "register_method",
]
