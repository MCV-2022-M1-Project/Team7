from abc import ABC
from typing import Any

from src.datasets.dataset import Dataset


class BaseTask(ABC):
    """
    Base task runner.
    """
    name: str

    def __init__(self, dataset: Dataset, config: Any, **kwargs) -> None:
        self.config = config
        self.dataset = dataset
        # self.preprocessing = preprocessing
        # self.metrics = [
        #     Registry.get_metric(name)() for name, distance in Registry.get("task").metrics
        # ]

    def run(self) -> None:
        """
        Something like this:
        for sample in self.dataset:
            image = sample.image

            for pp in self.preprocessing:
                image = pp.preprocess(image)

            features = self.extractor.run(image)
            # KNN stuff
            # Compute metrics and store in instance variables
            # so you can access them calling the object, i.e.
            # for metric in task.get_metrics():
            #   print(metric["name"], ":", metric["value"])
        """
        pass


