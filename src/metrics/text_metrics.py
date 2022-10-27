import numpy as np
import textdistance
from typing import List
from src.common.registry import Registry
from src.metrics.base import Metric


@Registry.register_metric
class LevenshteinMetric(Metric):
    name: str = "text_levenshtein"
    input_type: str = "str"
    
    def compute(self, ground_truth: List[List[str]], predictions: List[str]) -> float:
        val = 0.0

        # for preds, gts in zip(predictions, ground_truth):
        #     val += textdistance.levenshtein(preds, gts)

        for pred in predictions:
            max_val = 0

            for gt in ground_truth:
                for label in gt:
                    new_val = textdistance.levenshtein(pred, label)
                    max_val = max(max_val, new_val)

            val += max_val

        return val / min(len(ground_truth), len(predictions))


@Registry.register_metric
class EditexMetric(Metric):
    name: str = "text_editex"
    input_type: str = "str"
    
    def compute(self, ground_truth: List[List[str]], predictions: List[str]) -> float:
        val = 0.0

        # for preds, gts in zip(predictions, ground_truth):
        #     val += textdistance.editex(preds, gts)

        for pred in predictions:
            max_val = 0

            for gt in ground_truth:
                for label in gt:
                    new_val = textdistance.editex(pred, label)
                    max_val = max(max_val, new_val)

            val += max_val

        return val / min(len(ground_truth), len(predictions))


@Registry.register_metric
class JaccardMetric(Metric):
    name: str = "text_jaccard"
    input_type: str = "token"
    
    def compute(self, ground_truth: List[List[str]], predictions: List[str]) -> float:
        val = 0.0

        # for preds, gts in zip(predictions, ground_truth):
        #     val += textdistance.jaccard(preds, gts)

        for pred in predictions:
            max_val = 0

            for gt in ground_truth:
                for label in gt:
                    new_val = textdistance.jaccard(pred, label)
                    max_val = max(max_val, new_val)

            val += max_val

        return val / min(len(ground_truth), len(predictions))