import numpy as np
from numpy import array

from src.common.registry import Registry
from src.metrics.base import Metric
from src.common.utils import binarize, image_normalize


@Registry.register_metric
class MAE(Metric):
    name: str = "mae"

    def compute(self, ground_truth, predictions):
        """
            Compute the mean absolute error

            Args:
                'ground_truth': HxW or HxWxn (asumme that all the n channels are the same and only the first channel will be used)
                'predictions': HxW or HxWxn

            Returns: 
                a value MAE, Mean Absolute Error
        """
        ground_truth = array(ground_truth)
        predictions = array(predictions)

        if np.max(ground_truth) > 1.0:
            ground_truth = image_normalize(ground_truth)

        if np.max(predictions) > 1.0:
            predictions = image_normalize(predictions)

        h, w = ground_truth.shape[-2], ground_truth.shape[-1]
        sumError = np.sum(np.absolute(
            (ground_truth.astype(float) - predictions.astype(float))))
        maeError = sumError/((float(h)*float(w)+1e-8))
        return maeError


@Registry.register_metric
class Accuracy(Metric):
    name: str = "accuracy"

    def compute(self, ground_truth, predictions):
        """
            Compute the precision

            Args:
                'ground_truth': ground truth
                'predictions': predictions

            Returns: 
                 precision
        """
        ground_truth = array(ground_truth)
        predictions = array(predictions)
        ground_truth = binarize(ground_truth)
        predictions = binarize(predictions)
        accuracy = (predictions == ground_truth).sum() / \
            (np.prod(ground_truth.shape))
        return accuracy


@Registry.register_metric
class Precision(Metric):
    name: str = "precision"

    def compute(self, ground_truth, predictions):
        """
            Compute the precision

            Args:
                'ground_truth': ground truth
                'predictions': predictions

            Returns: 
                 precision
        """
        ground_truth = array(ground_truth)
        predictions = array(predictions)
        ground_truth = binarize(ground_truth)
        predictions = binarize(predictions)
        precision = ((predictions == 1) & (ground_truth == 1)
                     ).sum()/(predictions.sum()+1e-8)
        return precision


@Registry.register_metric
class Recall(Metric):
    name: str = "recall"

    def compute(self, ground_truth, predictions):
        """
            Compute the precision

            Args:
                'ground_truth': ground truth
                'predictions': predictions

            Returns: 
                 recall
        """
        ground_truth = array(ground_truth)
        predictions = array(predictions)
        ground_truth = binarize(ground_truth)
        predictions = binarize(predictions)
        recall = ((predictions == 1) & (ground_truth == 1)).sum() / \
            (ground_truth.sum()+1e-8)
        return recall


@Registry.register_metric
class F1(Metric):
    name: str = "f1"

    def compute(self, ground_truth, predictions):
        """
            Compute the f1

            Args:
                'ground_truth': ground truth
                'predictions': predictions

            Returns: 
                 f1
        """
        ground_truth = array(ground_truth)
        predictions = array(predictions)
        ground_truth = binarize(ground_truth)
        predictions = binarize(predictions)
        precision = ((predictions == 1) & (ground_truth == 1)
                     ).sum()/(predictions.sum()+1e-8)
        recall = ((predictions == 1) & (ground_truth == 1)).sum() / \
            (ground_truth.sum()+1e-8)
        return 2*recall*precision/(recall+precision+1e-5)


@Registry.register_metric
class IoU(Metric):
    name: str = "iou"
    input_type: str = "bb"

    def compute(self, ground_truth, predictions):
        # ground_truth = array(ground_truth, dtype=np.uint8)
        # predictions = array(predictions, dtype=np.uint8)

        total_iou = []

        for gt, pred in zip(ground_truth, predictions):
            for gt_text_box in gt:
                x11, y11, x12, y12 = gt_text_box
                max_iou = 0.0

                for pred_text_box in pred:
                    x21, y21, x22, y22 = pred_text_box
                    xa = max(x11, x21)
                    ya = max(y11, y21)
                    xb = min(x12, x22)
                    yb = min(y12, y22)
                    intersection = max(xb - xa, 0) * max(yb - ya, 0)
                    area_a = (x12-x11) * (y12-y11)
                    area_b = (x22-x21) * (y22-y21)
                    union = area_a + area_b - intersection
                    iou = intersection / union
                    max_iou = max(max_iou, iou)

                total_iou.append(max_iou)

        if len(total_iou) == 0:
            return 0

        return np.mean(total_iou)
