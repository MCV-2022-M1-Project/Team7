from typing import Dict, List

from datasets.dataset import Dataset
from preprocessing.base import Preprocessing


class Evaluator:
    """
    Datasets
    Preprocessing methods
    Extract features
    Metrics
    """
    def __init__(self, dataset: Dataset, preprocessing: List[Preprocessing], algorithm: FeaturesExtra) -> None:
        pass

    def extract_features(self) -> None:
        pass

    def compute_metrics(self, save: bool) -> Dict[str, float]:
        pass
