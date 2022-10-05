import numpy as np

from typing import Protocol, Dict


class FeaturesExtractor(Protocol):
    name: str
    
    def run(self, features: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        pass