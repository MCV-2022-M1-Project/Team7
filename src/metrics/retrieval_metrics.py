


from typing import List
from src.common.registry import Registry
from src.metrics.base import Metric


@Registry.register_metric
class RawAccuracyMetric(Metric):
    name: str = "raw_accuracy"
    
    def compute(self, ground_truth: List[List[int]], predictions: List[List[int]]) -> float:
        val = 0.0

        for i, gt in enumerate(ground_truth):
            if gt[0] == predictions[i][0]:
                val += 1.0

        return val / len(ground_truth)