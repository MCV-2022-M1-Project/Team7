import numpy as np

from typing import List, Protocol, Dict


class FeaturesExtractor(Protocol):
    name: str

    def __init__(self, *args, **kwargs) -> None:
        pass
    
    def run(self, images: List[np.ndarray], **kwargs) -> Dict[str, np.ndarray]:
        pass