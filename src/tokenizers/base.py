import numpy as np

from typing import Any, List, Dict, Protocol


class BaseTokenizer(Protocol):
    name: str

    def __init__(self, *args, **kwargs) -> None:
        pass

    def tokenize(self, sample) -> Dict[str, np.ndarray]:
        pass

    def fit(self, samples: List[Any], **kwargs) -> None:
        pass
