import numpy as np

from abc import ABC
from typing import List, Dict


class FeaturesExtractor(ABC):
    name: str
    returns_keypoints: bool = False 

    def __init__(self, *args, **kwargs) -> None:
        pass
    
    def run(self, images: List[np.ndarray], **kwargs) -> Dict[str, np.ndarray]:
        pass