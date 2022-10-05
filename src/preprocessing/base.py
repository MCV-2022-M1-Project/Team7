import numpy as np

from typing import Dict, Protocol


class Preprocessing(Protocol):
    name: str

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
    
    def run(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        pass