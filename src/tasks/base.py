import os
from abc import ABC
from typing import Any, Optional, List
from src.common.utils import MetricWrapper

from src.datasets.dataset import Dataset
from src.extractors.base import FeaturesExtractor
from src.metrics.base import Metric
from src.preprocessing.base import Preprocessing
from src.tokenizers.base import BaseTokenizer


class BaseTask(ABC):
    """
    Base task runner.
    """
    name: str

    def __init__(self,
                 retrieval_dataset: Dataset,
                 query_dataset: Dataset,
                 config: Any,
                 output_dir: str = "./output",
                 tokenizer: Optional[BaseTokenizer] = None,
                 preprocessing:  List[Preprocessing] = [],
                 features_extractor:  Optional[FeaturesExtractor] = None,
                 metrics: List[MetricWrapper] = [],
                 id: str = '0',
                 *args,
                 **kwargs
                 ) -> None:
        self.id = id
        self.config = config
        self.retrieval_dataset = retrieval_dataset
        self.query_dataset = query_dataset
        self.output_dir = os.path.join(
            output_dir, f"{self.name}_on_{self.query_dataset.name}_{id}")
        self.report_path = os.path.join(self.output_dir, "report.txt")
        os.makedirs(self.output_dir, exist_ok=True)
        self.tokenizer = tokenizer
        self.extractor = features_extractor
        self.preprocessing = preprocessing
        self.metrics = metrics

    def run(self, inference_only: bool = False,  *args, **kwargs) -> None:
        """
        Method to run the task.

        Args:
            inference_only: whether to evaluate the results or not
        """
        pass
