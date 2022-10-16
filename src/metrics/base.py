from abc import ABC
import numpy as np

from typing import Any, Protocol, List


class Metric(ABC):
    name: str

    def __init__(self, *args, **kwargs) -> None:
        pass
    
    def compute(self, ground_truth: Any, predictions: Any, **kwargs) -> float:
        pass

    def __str__(self) -> str:
        return self.name


class GraphMetric(Protocol):
    name: str
    
    def compute(self, ground_truth: Any, predictions: Any, **kwargs) -> None:
        pass
