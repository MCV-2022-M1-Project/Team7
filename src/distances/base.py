from abc import ABC

from typing import Any


class BaseDistance(ABC):
    name: str
    input_type: str = "any"

    def __init__(self, *args, **kwargs) -> None:
        pass
    
    def compute(self, X: Any, Y: Any, **kwargs) -> float:
        pass
    
    def get_reference(self) -> Any:
        """
        Returns either a str if it's already implemented in sklearn
        or a Callable (compute() function) that receives two vectors.

        For example:
        return "l1"
        # or
        return self.compute
        # or
        return library.function
        """
        pass