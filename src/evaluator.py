from typing import Dict, List

from src.datasets.dataset import Dataset
from src.extractors.base import FeaturesExtractor
from src.preprocessing.base import Preprocessing


class Evaluator:
    """
    Datasets
    Preprocessing methods
    Extract features
    Metrics
    """
    def __init__(self, dataset: Dataset, preprocessing: List[Preprocessing], algorithm: FeaturesExtractor) -> None:
        pass

    def evaluate(self) -> None:
        pass
