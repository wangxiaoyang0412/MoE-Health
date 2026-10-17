"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["set_seed"]


def set_seed(seed: int) -> torch.Generator:
    """Seed python/numpy/torch and return a CPU generator for batch shuffling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen
