"""Data module: cohort construction, feature extraction helpers and the training dataset."""

from src.data_module.dataset import DataConfig, SplitData, TaskData, load_task_data
from src.data_module.registry import DATASET_FACTORY, DatasetFactory, register_dataset

__all__ = [
    "DATASET_FACTORY",
    "DataConfig",
    "DatasetFactory",
    "SplitData",
    "TaskData",
    "load_task_data",
    "register_dataset",
]
