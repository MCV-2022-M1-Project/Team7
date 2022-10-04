import numpy as np

from typing import Dict, Protocol


class Preprocessing(Protocol):    
    def preprocess(features: np.ndarray) -> Dict[str, np.ndarray]:
        pass