"""
Registry serves as a global storage for all configs that are used in the project.

Usage:
from src.common.registry import Registry

# Register a config
Registry.register("trainer_config", trainer_config)
Registry.register("model_config", model_config)
Registry.register("dataset_config", dataset_config)

# Access a config
trainer_config = Registry.get("trainer_config")
model_config = Registry.get("model_config")
dataset_config = Registry.get("dataset_config")
"""

from typing import Any, Dict

from src.preprocessing.base import Preprocessing
from src.datasets.dataset import Dataset
from src.extractors.base import FeaturesExtractor
from src.metrics.base import Metric


class Registry:
    _registry: Dict[str, Any] = {}
    _preprocessing: Dict[str, Preprocessing] = {}
    _features_extractors: Dict[str, FeaturesExtractor] = {}
    _datasets: Dict[str, Dataset] = {}
    _metrics: Dict[str, Metric] = {}

    @classmethod
    def register(cls, name: str, config: Any) -> None:
        cls._registry[name] = config

    @classmethod
    def register_preprocessing(cls, cl) -> None:
        cls._preprocessing[cl.name] = cl

    @classmethod
    def register_features_extractor(cls, cl) -> None:
        cls._features_extractors[cl.name] = cl

    @classmethod
    def register_dataset(cls, cl) -> None:
        cls._datasets[cl.name] = cl

    @classmethod
    def register_metric(cls, cl) -> None:
        cls._metrics[cl.name] = cl

    @classmethod
    def get(cls, name: str) -> Any:
        return cls._registry[name]

    @classmethod
    def get_preprocessing(cls, name: str) -> Preprocessing:
        return cls._preprocessing[name]

    @classmethod
    def get_features_extractor(cls, name: str) -> FeaturesExtractor:
        return cls._features_extractors[name]

    @classmethod
    def get_dataset(cls, name: str) -> Dataset:
        return cls._datasets[name]

    @classmethod
    def get_metric(cls, name: str) -> Metric:
        return cls._metrics[name]
