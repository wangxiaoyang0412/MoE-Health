"""Chest X-ray image loading in the torchxrayvision convention."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchvision
import torchxrayvision as xrv
from PIL import Image
from torch.utils.data import Dataset

__all__ = ["CXRImages"]


class CXRImages(Dataset):
    """Grayscale JPG -> [-1024, 1024] -> center crop -> 224 x 224."""

    def __init__(self, root: Path, rel_paths: list[str]) -> None:
        self.root = root
        self.paths = rel_paths
        self.transform = torchvision.transforms.Compose(
            [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        img = np.asarray(Image.open(self.root / self.paths[i]).convert("L"), dtype=np.float32)
        img = xrv.datasets.normalize(img, 255)[None, :, :]
        return torch.from_numpy(self.transform(img).astype(np.float32))
