import itertools
from typing import Any, Dict, List, Type, Optional

from src.preprocessing.base import Preprocessing
from src.datasets.dataset import Dataset
from src.extractors.base import FeaturesExtractor
from src.metrics.base import Metric
from src.tasks.base import BaseTask
from src.tokenizers.base import BaseTokenizer
from src.distances.base import BaseDistance


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
    _tokenizers: Dict[str, Type[BaseTokenizer]] = {}
    _distances: Dict[str, Type[BaseDistance]] = {}

    @classmethod
    def register(cls, name: str, value: Any) -> None:
        cls._registry[name] = value

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
    def register_tokenizer(cls, cl) -> Type[BaseTokenizer]:
        cls._tokenizers[cl.name] = cl
        return cl

    @classmethod
    def register_distance(cls, cl) -> Type[BaseDistance]:
        cls._distances[cl.name] = cl
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
    def get_task_class(cls, name: str) -> Type[BaseTask]:
        if name not in cls._tasks:
            raise Exception(f"Task '{name}' not registered. Available options are: {', '.join(cls._tasks)}")

        return cls._tasks[name]

    @classmethod
    def get_tokenizer_instance(cls, tokenizer_config: Any) -> BaseTokenizer:
        name = tokenizer_config.name

        if name not in cls._tokenizers:
            raise Exception(f"Tokenizer '{name}' not registered. Available options are: {', '.join(cls._tokenizers)}")

        return cls._tokenizers[name](**tokenizer_config)

    @classmethod
    def get_distance_instance(cls, distance_config: Any) -> BaseDistance:
        name = distance_config.name

        if name not in cls._distances:
            raise Exception(f"Distance '{name}' not registered. Available options are: {', '.join(cls._distances)}")

        return cls._distances[name](**distance_config)

    @classmethod
    def get_features_extractor_instances(cls, features_extractor_config: Any) -> List[FeaturesExtractor]:
        return [
            cls.get_features_extractor_class(kwargs["name"])(**kwargs) for kwargs in features_extractor_config
        ]

    @classmethod
    def get_preprocessing_instances(cls, preprocessing_configs: Any) -> List[Preprocessing]:
        return [
            cls.get_preprocessing_class(kwargs["name"])(**kwargs) for kwargs in preprocessing_configs
        ]

    @classmethod
    def get_metric_instances(cls, metrics_configs: Any) -> List[Metric]:
        metrics = []

        for mc in metrics_configs:
            m_class = cls.get_metric_class(mc["name"])
            params_list = []
            num_params = 0

            for item, val in mc.items():
                if item == "name":
                    continue 
                
                num_params += 1

                if type(val) is list:
                    params_list += [(item, v) for v in val]
                else:
                    params_list.append((item, val))

            if num_params == 0:
                metrics.append(m_class())
                continue

            param_combinations = list(itertools.combinations(params_list, num_params))

            for param_combination in param_combinations:
                pcd = dict(param_combination)

                if len(pcd) < len(mc) - 1:
                    continue

                metrics.append(m_class(**pcd))

        return metrics

    @classmethod
    def get_datasets(cls) -> Dict[str, Dataset]:
        return cls._datasets

