import numpy as np
from typing import List
from src.common.registry import Registry
from src.metrics.base import Metric


@Registry.register_metric
class RawAccuracyMetric(Metric):
    name: str = "raw_accuracy"
    
    def compute(self, ground_truth: List[List[int]], predictions: List[List[List[int]]]) -> float:
        val = 0.0

        for preds, gts in zip(predictions, ground_truth):
            for painting in preds:
                if painting[0] in gts:
                    val += 1.0

        return val / len(ground_truth)


@Registry.register_metric
class MAP(Metric):
    name: str = "map"

    def __init__(self, top_k: int = 10, *args, **kwargs) -> None:
        self.top_k = top_k
    
    def apk(self, ground_truth: int, predictions: List[int]) -> float:

        """
        Computes the average precision at k.
        This function computes the average prescision at k between two lists of
        items.

        Taken from: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py

        Args:
            actual : list
                 A list of elements that are to be predicted (order doesn't matter)
            predicted : list
                A list of predicted elements (order does matter)
            k : int, optional
                The maximum number of predicted elements

        Returns:
            score : double
            The average precision at k over the input lists
        """
        if not ground_truth:
            return 0.0

        if len(predictions) > self.top_k:
            predictions = predictions[:self.top_k]

        score = 0.0
        num_hits = 0.0
        relevant_docs = 0

        for i,p in enumerate(predictions):
            # first condition checks whether it is valid prediction
            # second condition checks if prediction is not repeated
            if p == ground_truth and p not in predictions[:i]:
                num_hits += 1.0
                relevant_docs += 1
                score += num_hits / (i+1.0)

        return score # / min(len(ground_truth), 10)

    def compute(self, ground_truth: List[int], predictions: List[List[int]]) -> float:
        return np.mean([self.apk(a,p) for a,p in zip(ground_truth, predictions)])

    def __str__(self) -> str:
        return f"{self.name}[@{self.top_k}]"
    

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

