import numpy as np

from typing import Dict, Protocol


class Preprocessing(Protocol):
    name: str
    
    def run(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        pass