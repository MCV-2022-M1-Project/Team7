import numpy as np

from typing import Protocol, Dict


class FeaturesExtractor(Protocol):
    name: str
    
    def run(features: np.ndarray) -> Dict[str, np.ndarray]:
        pass