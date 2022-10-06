import numpy as np


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
    
@Registry.register_metric
class EuclDis(Metric):
    name: str = "euclidian_distance"
    
    def compute(self, ground_truth: List[List[int]], predictions: List[List[int]]) -> float:
        dis = np.sum([(a - b) ** 2
        for (a, b) in zip(ground_truth, predictions)])
        return np.sqrt(dis)
    
@Registry.register_metric
class L1Dis(Metric):
    name: str = "l1_distance"
    
    def compute(self, ground_truth: List[List[int]], predictions: List[List[int]]) -> float:
        d = np.sum([abs(a - b) for (a, b) in zip(ground_truth, predictions)])
        return d
    
@Registry.register_metric
class Chi2Dis(Metric):
    name: str = "chi2_distance"
    
    def compute(self, ground_truth: List[List[int]], predictions: List[List[int]]) -> float:
        d = np.sum([((a - b) ** 2) / (a + b + 1e-8)
            for (a, b) in zip(ground_truth, predictions)])
        return d
    
@Registry.register_metric
class HellKDis(Metric):
    name: str = "hellinger_kernel_distance"
    
    def compute(self, ground_truth: List[List[int]], predictions: List[List[int]]) -> float:
        d = np.sum([np.sqrt(a*b)
            for (a, b) in zip(ground_truth, predictions)])
        return d
@Registry.register_metric
class HistIntSim(Metric):
    name: str = "histogram_intersection_similarity"
    
    def compute(self, ground_truth: List[List[int]], predictions: List[List[int]]) -> float:
            sm = 0
            ground_truth = np.array(ground_truth)
            predictions = np.array(predictions)
            for i in range(5):
                sm += min(ground_truth.all(), predictions.all())
            return sm
