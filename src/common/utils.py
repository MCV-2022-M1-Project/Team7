import json
import numpy as np
from dataclasses import dataclass, field
from typing import Any, List, Optional

from src.metrics.base import Metric


@dataclass
class MetricWrapper:
    """
    This class provides an interface to store the Metric results
    of an epoch.
    """
    metric: Metric
    values: List[float] = field(default_factory=list)
    running_total: float = 0.0
    num_updates: float = 0.0
    average: float = 0.0

    def update(self, value: float, batch_size: int = 1) -> None:
        self.values.append(value)
        self.running_total += value * batch_size
        self.num_updates += batch_size
        self.average = self.running_total / self.num_updates

    def compute(self, ground_truth: Any, predictions: Any, batch_size: int = 1) -> float:
        result = self.metric.compute(ground_truth, predictions)
        self.update(result, batch_size)
        return result


def wrap_metric_classes(metrics_list: List[Metric]) -> List[MetricWrapper]:
    return [MetricWrapper(metric) for metric in metrics_list]


def image_normalize(img: np.ndarray) -> np.ndarray:
    """
    Args:
        img: HxW [0, 255]

    Output: 
        HxW [0, 1]
    """ 
    return img / 255


def binarize(img: np.ndarray) -> np.ndarray:
    """
    Args:
        img: HxW

    Output: 
        HxW [0,255]
    """
    return img != 0


def write_report(path: str, task_config: Any, metrics: Optional[List[MetricWrapper]] = None):
    content = "Task configuration: \n\n"
    content += json.dumps(task_config, indent=2)

    if metrics is not None:
        content += f"\n\n--- Metrics ---\n"
        
        for metric in metrics:
            content += f"{metric.metric.name}: {metric.average}\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

        
def compute_pre_rec(gt,mask,mybins=np.arange(0,256)):
    """
        Auxiliary function for computing precision and recall
        
        Args:
            gt: ground-truth mask
            mask: predicted mask
        output:
            confusion matrix
    """

    if(len(gt.shape)<2 or len(mask.shape)<2):
        print("ERROR: gt or mask is not matrix!")
        exit()
    if(len(gt.shape)>2): # convert to one channel
        gt = gt[:,:,0]
    if(len(mask.shape)>2): # convert to one channel
        mask = mask[:,:,0]
    if(gt.shape!=mask.shape):
        print("ERROR: The shapes of gt and mask are different!")
        exit()

    gtNum = gt[gt>128].size # pixel number of ground truth foreground regions
    pp = mask[gt>128] # mask predicted pixel values in the ground truth foreground region
    nn = mask[gt<=128] # mask predicted pixel values in the ground truth bacground region

    pp_hist,pp_edges = np.histogram(pp,bins=mybins) #count pixel numbers with values in each interval [0,1),[1,2),...,[mybins[i],mybins[i+1]),...,[254,255)
    nn_hist,nn_edges = np.histogram(nn,bins=mybins)

    pp_hist_flip = np.flipud(pp_hist) # reverse the histogram to the following order: (255,254],...,(mybins[i+1],mybins[i]],...,(2,1],(1,0]
    nn_hist_flip = np.flipud(nn_hist)

    pp_hist_flip_cum = np.cumsum(pp_hist_flip) # accumulate the pixel number in intervals: (255,254],(255,253],...,(255,mybins[i]],...,(255,0]
    nn_hist_flip_cum = np.cumsum(nn_hist_flip)

    precision = pp_hist_flip_cum/(pp_hist_flip_cum + nn_hist_flip_cum+1e-8) #TP/(TP+FP)
    recall = pp_hist_flip_cum/(gtNum+1e-8) #TP/(TP+FN)

    precision[np.isnan(precision)]= 0.0
    recall[np.isnan(recall)] = 0.0

    return np.reshape(precision,(len(precision))),np.reshape(recall,(len(recall)))
