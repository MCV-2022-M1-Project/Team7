import numpy as np

from typing import Dict, Protocol


class Preprocessing(Protocol):
    name: str
    
    def preprocess(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        pass