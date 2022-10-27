import textdistance
from typing import Any
from src.common.registry import Registry
from src.metrics.base import Metric


@Registry.register_distance
class CosineMetric(Metric):
    name: str = "cosine"
    input_type: str = "any"
    
    def get_reference(self) -> str:
        return "cosine"


@Registry.register_distance
class L1Metric(Metric):
    name: str = "l1"
    input_type: str = "any"
    
    def get_reference(self) -> str:
        return "l1"