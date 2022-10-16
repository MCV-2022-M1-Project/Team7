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
    
    def _map(self, probabilities: List[float], labels: List[bool], k: int) -> float:

        """
        Expects: Probabilities (0, 1)
        Labels Bool (relevant / not relevant)
        """
        buffer = 0
        for at_k in range(1, k+1):
            for p, rel in zip(probabilities[:at_k], labels):
                buffer += (p*rel)/sum(labels) # Beware, labels should be K lenght
        return buffer / k
    
    def apk(self, ground_truth: List[int], predictions: List[int], k: int = 10) -> float:

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

        score = []

        for gt in ground_truth:
            max_score = 0.0

            for painting_preds in predictions:
                new_score = 0.0
                num_hits = 0.0

                for i, p in enumerate(painting_preds[:self.top_k]):
                    # first condition checks whether it is valid prediction
                    # second condition checks if prediction is not repeated
                    if p == gt and p not in painting_preds[:i]:
                        num_hits += 1.0
                        new_score += num_hits / (i+1.0)

                if new_score > max_score:
                    max_score = new_score

            score.append(max_score)

        return float(np.mean(score))

    def compute(self, ground_truth: List[List[int]], predictions: List[List[List[int]]], k: int = 10) -> float:
        return np.mean([self.apk(a,p,k) for a,p in zip(ground_truth, predictions)])

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

