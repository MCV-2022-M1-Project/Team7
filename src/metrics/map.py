# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 11:20:25 2022

@author: Ayan
"""

import numpy as np


from typing import List
from src.common.registry import Registry
from src.metrics.base import Metric

@Registry.register_metric
class MAP(Metric):
    name: str = "map"
    
    def compute(self, ground_truth: List[List[int]], predictions: List[List[int]]) -> float:

        """
        Computes the average precision at k.
        This function computes the average prescision at k between two lists of
        items.
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
        ground_truth = np.array(ground_truth)
        ground_truth = list(ground_truth.ravel())
        predictions = np.array(predictions)
        predictions = list(predictions.ravel())
        if not ground_truth:
            return 0.0

        if len(predictions)>10:
            predictions = predictions[:10]

        score = 0.0
        num_hits = 0.0

        for i,p in enumerate(predictions):
            # first condition checks whether it is valid prediction
            # second condition checks if prediction is not repeated
            if p in ground_truth and p not in predictions[:i]:
                num_hits += 1.0
                score += num_hits / (i+1.0)
        return score / min(len(ground_truth), 10)