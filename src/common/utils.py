from dataclasses import dataclass, field
from typing import Any, List

import numpy as np

from src.metrics.base import Metric


@dataclass
class MetricWrapper:
    """
    This class provides an interface to store the Metric results
    of an epoch.
    """
    metric: Metric
    values: List[float] = field(default_factory=list)
    running_total: float = 0.0
    num_updates: float = 0.0
    average: float = 0.0

    def update(self, value: float, batch_size: int = 1) -> None:
        self.values.append(value)
        self.running_total += value * batch_size
        self.num_updates += batch_size
        self.average = self.running_total / self.num_updates

    def compute(self, ground_truth: Any, predictions: Any, batch_size: int = 1) -> float:
        result = self.metric.compute(ground_truth, predictions)
        self.update(result)
        return result


def wrap_metric_classes(metrics_list: List[Metric]) -> List[MetricWrapper]:
    return [MetricWrapper(metric) for metric in metrics_list]


def image_normalize(img):
    """
    Args:
        mask: HxW

    Output: 
        HxW [0,255]
    """
    return img/(np.amax(img)+1e-7)
