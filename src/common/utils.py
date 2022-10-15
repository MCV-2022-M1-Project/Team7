import json
import numpy as np
import cv2
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


def compute_pre_rec(gt, mask, mybins=np.arange(0, 256)):
    """
        Auxiliary function for computing precision and recall

        Args:
            gt: ground-truth mask
            mask: predicted mask
        output:
            confusion matrix
    """

    if(len(gt.shape) < 2 or len(mask.shape) < 2):
        print("ERROR: gt or mask is not matrix!")
        exit()
    if(len(gt.shape) > 2):  # convert to one channel
        gt = gt[:, :, 0]
    if(len(mask.shape) > 2):  # convert to one channel
        mask = mask[:, :, 0]
    if(gt.shape != mask.shape):
        print("ERROR: The shapes of gt and mask are different!")
        exit()

    # pixel number of ground truth foreground regions
    gtNum = gt[gt > 128].size
    # mask predicted pixel values in the ground truth foreground region
    pp = mask[gt > 128]
    # mask predicted pixel values in the ground truth bacground region
    nn = mask[gt <= 128]

    # count pixel numbers with values in each interval [0,1),[1,2),...,[mybins[i],mybins[i+1]),...,[254,255)
    pp_hist, pp_edges = np.histogram(pp, bins=mybins)
    nn_hist, nn_edges = np.histogram(nn, bins=mybins)

    # reverse the histogram to the following order: (255,254],...,(mybins[i+1],mybins[i]],...,(2,1],(1,0]
    pp_hist_flip = np.flipud(pp_hist)
    nn_hist_flip = np.flipud(nn_hist)

    # accumulate the pixel number in intervals: (255,254],(255,253],...,(255,mybins[i]],...,(255,0]
    pp_hist_flip_cum = np.cumsum(pp_hist_flip)
    nn_hist_flip_cum = np.cumsum(nn_hist_flip)

    precision = pp_hist_flip_cum / \
        (pp_hist_flip_cum + nn_hist_flip_cum+1e-8)  # TP/(TP+FP)
    recall = pp_hist_flip_cum/(gtNum+1e-8)  # TP/(TP+FN)

    precision[np.isnan(precision)] = 0.0
    recall[np.isnan(recall)] = 0.0

    return np.reshape(precision, (len(precision))), np.reshape(recall, (len(recall)))


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def compute_iou(m1, m2):
    m1 = m1.astype(np.uint8)
    m2 = m2.astype(np.uint8)
    overlap = (m1 != 0) & (m2 != 0)  # Logical AND
    union = (m1 != 0) | (m2 != 0)  # Logical OR
    return overlap.sum() / float(union.sum())


def generate_binary_mask(mask):
    """
    Binarizes a given mask that can be in RGB format or not normalized.

    Parameters
    ----------
    mask : mask to binarize

    Returns
    -------
    mask : binarized and normalized mask (0 and 1)
    """

    # if mask is RGB => to grayscale
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # if mask is not binary, we transform it
    if not(np.unique(mask) == (0, 1)).all():
        mask = (mask == 255).astype(float)
    return mask


def opening(m, size=(45, 45)):
    kernel_v = np.ones((size[0], 1), np.uint8)
    kernel_h = np.ones((1, size[1]), np.uint8)
    m = cv2.erode(m, kernel_v, iterations=1)
    m = cv2.erode(m, kernel_h, iterations=1)
    m = cv2.dilate(m, kernel_v, iterations=1)
    m = cv2.dilate(m, kernel_h, iterations=1)
    return m


def closing(m, size=(45, 45)):
    kernel_v = np.ones((size[0], 1), np.uint8)
    kernel_h = np.ones((1, size[1]), np.uint8)
    m = cv2.dilate(m, kernel_v, iterations=1)
    m = cv2.dilate(m, kernel_h, iterations=1)
    m = cv2.erode(m, kernel_v, iterations=1)
    m = cv2.erode(m, kernel_h, iterations=1)
    return m


def gray_historam(image: np.ndarray, bins: int = 256, mask: np.ndarray = None) -> np.ndarray:
    """
    Extract histogram from grascale version of image

    Args:
        image: (H x W x C) 3D BGR image array of type np.uint8
        bins: number of bins to use for histogram
        mask: check _descriptor(first function in file)

    Returns: 
        1D  array of type np.float32 containing histogram
        feautures of image
    """
    print(mask is None)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_image], [0], mask, [bins], [0, 256])
    hist = cv2.normalize(hist, hist)
    return hist.flatten()


def extract_biggest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    Extracts the biggest connected component from a mask (0 and 1's).
    
    Args:
        img: 2D array of type np.float32 representing the mask

    Returns : 2D array, mask with 1 in the biggest component and 0 outside
    """
    # extract all connected components
    num_labels, labels_im = cv2.connectedComponents(mask.astype(np.uint8))

    # we find and return only the biggest one
    max_val, max_idx = 0, -1

    for i in range(1, num_labels):
        area = np.sum(labels_im == i)

        if area > max_val:
            max_val = area
            max_idx = i

    return (labels_im == max_idx).astype(float)


def extract_paintings_from_mask(mask: np.ndarray):
    to_return = []
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))

    for lab in range(1, num_labels):
        m = (labels == lab).astype(np.uint8)
        first_pixel = np.min(np.where(m != 0)[1])
        to_return.append((m, first_pixel))
        
    both = list(zip(*sorted(to_return, key=lambda t: t[1])))
    return both[0]


def generate_text_mask(shape, textboxes):
    if textboxes is None or len(textboxes) == 0:
        return np.zeros(shape).astype(np.uint8)

    mask = np.zeros(shape)

    for (xtl, ytl, xbr, ybr) in textboxes:
        pts = np.array(((xtl, ytl), (xtl, ybr), (xbr, ybr), (xbr, ytl)))
        cv2.fillConvexPoly(mask, pts, True)

    return mask.astype(np.uint8)


def evaluate_textboxes(gt_boxes, boxes):
    """
    Evaluates the mean intersection-over-union between GT textboxes and the given ones.

    Parameters
    ----------
    gt_boxes : list of textboxes for each image, as described in W2 slides
    boxes : list of textboxes for each image, as described in W2 slides

    Returns
    -------
    mean_iou: mean intersection-over-union
    """
    assert len(gt_boxes) == len(boxes)

    iou = 0
    # compute IOU per image
    for i in range(len(boxes)):
        if len(boxes[i]) == 0 or len(gt_boxes[i]) == 0:
            continue

        max_dim = np.max(np.max(boxes[i]))
        shape = (max_dim, max_dim)
        # We compute the IOU by generating both masks with all given textboxes highlighted.
        gt_mask, mask = generate_text_mask(
            shape, gt_boxes[i]), generate_text_mask(shape, boxes[i])
        iou += compute_iou(gt_mask, mask)
    return iou / len(boxes)
