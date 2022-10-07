import numpy as np

from typing import List, Protocol, Dict


class BaseTokenizer(Protocol):
    name: str

    def __init__(self, *args, **kwargs) -> None:
        pass

    def tokenize(self, sample) -> Dict[str, np.ndarray]:
        pass

    def fit(self, samples: List[np.ndarray], **kwargs) -> None:
        pass
