import numpy as np

from typing import Any, Protocol, List


class Metric(Protocol):
    name: str
    
    def compute(self, ground_truth: Any, predictions: Any, **kwargs) -> float:
        pass


class GraphMetric(Protocol):
    name: str
    
    def compute(self, ground_truth: Any, predictions: Any, **kwargs) -> None:
        pass
