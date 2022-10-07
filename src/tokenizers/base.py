import numpy as np

from typing import List, Protocol, Dict


class Tokenizer(Protocol):
    name: str
    def tokenize(self, sample):
        pass
    def fit(self, samples: List[np.ndarray], **kwargs) -> Dict[str, np.ndarray]:
        pass