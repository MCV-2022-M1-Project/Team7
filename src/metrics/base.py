from abc import ABC
import numpy as np

from typing import Any, Protocol, List


class Metric(ABC):
    name: str
    input_type: str = "any"

    def __init__(self, *args, **kwargs) -> None:
        pass
    
    def compute(self, ground_truth: Any, predictions: Any, **kwargs) -> float:
        pass

    def __str__(self) -> str:
        return self.name