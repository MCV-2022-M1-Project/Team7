import textdistance
from typing import Callable
from src.common.registry import Registry
from src.metrics.base import Metric


@Registry.register_distance
class JaccardDistance(Metric):
    name: str = "jaccard_dist"
    input_type: str = "token"
    
    def get_reference(self) -> Callable:
        return textdistance.jaccard


@Registry.register_distance
class BagDistance(Metric):
    name: str = "bag_dist"
    input_type: str = "token"
    
    def get_reference(self) -> Callable:
        return textdistance.bag


