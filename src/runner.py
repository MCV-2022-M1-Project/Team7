from typing import Dict, List

from src.datasets.dataset import Dataset
from src.extractors.base import FeaturesExtractor
from src.preprocessing.base import Preprocessing


class Runner:
    """
    Datasets
    Preprocessing methods
    Extract features
    Metrics
    """
    def __init__(self, dataset: Dataset, preprocessing: List[Preprocessing], extractor: FeaturesExtractor) -> None:
        self.dataset = dataset
        self.preprocessing = preprocessing
        self.extractor = extractor

    def run(self) -> None:
        for sample in self.dataset:
            image = sample.image

            for pp in self.preprocessing:
                image = pp.preprocess(image)

            features = self.extractor.run(image)


