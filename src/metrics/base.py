import numpy as np

from typing import Protocol, List


class Metric(Protocol):
    name: str
    
    def compute(self, features: np.ndarray) -> List[float]:
        pass