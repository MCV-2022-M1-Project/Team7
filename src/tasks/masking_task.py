from abc import ABC
import logging

from src.common.registry import Registry
from src.common.utils import wrap_metric_classes
from src.datasets.dataset import Dataset
from src.extractors.base import FeaturesExtractor
from src.preprocessing.base import Preprocessing
from src.tasks.base import BaseTask


class MaskingTask(BaseTask):
    """
    Base task runner.
    """
    name: str = "masking"

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.preprocessing = Registry.get_selected_preprocessing_instances()
        self.metrics = Registry.get_selected_metric_instances()
        self.metrics = wrap_metric_classes(self.metrics)

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
        for sample in self.dataset:
            image = sample.image
            mask_gt = sample.mask

            for pp in self.preprocessing:
                output = pp.run(image)
                image = output["result"]

            mask_pred = image

            for metric in self.metrics:
                metric.compute(mask_gt, mask_pred)

        for metric in self.metrics:
            logging.info(f"{metric.metric.name}: {metric.average}")

            # KNN stuff
            # Compute metrics and store in instance variables
            # so you can access them calling the object, i.e.
            # for metric in task.get_metrics():
            #   print(metric["name"], ":", metric["value"])