from typing import Any, Dict, List, Type

from src.preprocessing.base import Preprocessing
from src.datasets.dataset import Dataset
from src.extractors.base import FeaturesExtractor
from src.metrics.base import Metric
from src.tasks.base import BaseTask


class Registry:
    """
    Registry serves as a global storage for all configs that are used in the project.

    Usage:
    from src.common.registry import Registry

    # Register a config or a class
    Registry.register("example_config", config_dict)
    Registry.register_preprocessing
    ...

    # Access a config
    example_config = Registry.get("example_config")
    example = Registry.get_preprocessing("example_preprocessing_method")
    ...
    """
    _registry: Dict[str, Any] = {}
    _preprocessing: Dict[str, Type[Preprocessing]] = {}
    _features_extractors: Dict[str, Type[FeaturesExtractor]] = {}
    _datasets: Dict[str, Dataset] = {}
    _metrics: Dict[str, Type[Metric]] = {}
    _tasks: Dict[str, Type[BaseTask]] = {}

    @classmethod
    def register(cls, name: str, config: Any) -> None:
        cls._registry[name] = config

    @classmethod
    def register_preprocessing(cls, cl) -> Type[Preprocessing]:
        cls._preprocessing[cl.name] = cl
        return cl

    @classmethod
    def register_features_extractor(cls, cl) -> Type[FeaturesExtractor]:
        cls._features_extractors[cl.name] = cl
        return cl

    @classmethod
    def register_dataset(cls, name: str, dataset: Dataset) -> None:
        cls._datasets[name] = dataset

    @classmethod
    def register_metric(cls, cl) -> Type[Metric]:
        cls._metrics[cl.name] = cl
        return cl

    @classmethod
    def register_task(cls, cl) -> Type[BaseTask]:
        cls._tasks[cl.name] = cl
        return cl

    @classmethod
    def get(cls, name: str) -> Any:
        return cls._registry[name]

    @classmethod
    def get_preprocessing_class(cls, name: str) -> Type[Preprocessing]:
        if name not in cls._preprocessing:
            raise Exception(f"Preprocessing class '{name}' not registered. Available options are: {', '.join(cls._preprocessing)}")

        return cls._preprocessing[name]

    @classmethod
    def get_features_extractor_class(cls, name: str) -> Type[FeaturesExtractor]:
        if name not in cls._features_extractors:
            raise Exception(f"Feature extractor '{name}' not registered. Available options are: {', '.join(cls._features_extractors)}")

        return cls._features_extractors[name]

    @classmethod
    def get_dataset_class(cls, name: str) -> Dataset:
        if name not in cls._datasets:
            raise Exception(f"Dataset '{name}' not registered. Available options are: {', '.join(cls._datasets)}")

        return cls._datasets[name]

    @classmethod
    def get_metric_class(cls, name: str) -> Type[Metric]:
        if name not in cls._metrics:
            raise Exception(f"Metric '{name}' not registered. Available options are: {', '.join(cls._metrics)}")

        return cls._metrics[name]

    @classmethod
    def get_selected_task_class(cls) -> Type[BaseTask]:
        name = Registry.get("task").name

        if name not in cls._tasks:
            raise Exception(f"Task '{name}' not registered. Available options are: {', '.join(cls._tasks)}")

        return cls._tasks[name]

    @classmethod
    def get_selected_features_extractor_instance(cls) -> FeaturesExtractor:
        return cls.get_features_extractor_class(Registry.get("task").features_extractor.name)() 

    @classmethod
    def get_selected_preprocessing_instances(cls) -> List[Preprocessing]:
        return [
            cls.get_preprocessing_class(kwargs["name"])(**kwargs) for kwargs in Registry.get("task").preprocessing
        ]

    @classmethod
    def get_selected_metric_instances(cls) -> List[Metric]:
        return [
            cls.get_metric_class(m["name"])() for m in Registry.get("task").metrics
        ]

    @classmethod
    def get_datasets(cls) -> Dict[str, Dataset]:
        return cls._datasets

