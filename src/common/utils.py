import json
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Any, List, Optional
from scipy.signal import convolve2d

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


def image_resize( image, width = None, height = None, inter = cv2.INTER_AREA):
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
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def compute_iou(m1, m2):
    m1 = m1.astype(np.uint8)
    m2 = m2.astype(np.uint8)
    overlap = (m1 != 0) & (m2 != 0) # Logical AND
    union = (m1 != 0) | (m2 != 0) # Logical OR
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
    kernel = np.ones(size, np.uint8) 
    m = cv2.erode(m, kernel, iterations=1) 
    m = cv2.dilate(m, kernel, iterations=1) 
    return m

def closing(m, size=(45, 45)):
    kernel = np.ones(size, np.uint8) 
    m = cv2.dilate(m, kernel, iterations=1) 
    m = cv2.erode(m, kernel, iterations=1) 
    return m

def gray_historam(image:np.ndarray, bins:int=256, mask:np.ndarray=None) -> np.ndarray:
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
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_image],[0], mask, [bins], [0,256])
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

def extract_paintings_from_mask(mask:np.ndarray):
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
        gt_mask, mask = generate_text_mask(shape, gt_boxes[i]), generate_text_mask(shape, boxes[i])
        iou += compute_iou(gt_mask, mask)
    return iou / len(boxes)

def brightText(img):
    """
    Generates the textboxes candidated based on TOPHAT morphological filter.
    Works well with bright text over dark background.
    
    Parameters
    ----------
    img : ndimage to process
    
    Returns
    -------
    mask: uint8 mask with regions of interest (possible textbox candidates)
    """
    kernel = np.ones((30, 30), np.uint8) 
    img_orig = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    
    TH = 150
    img_orig[(img_orig[:,:,0] < TH) | (img_orig[:,:,1] < TH) | (img_orig[:,:,2] < TH)] = (0,0,0)
    
    img_orig = closing(img_orig, size=(1, int(img.shape[1] / 8)))
    
    return (cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY) != 0).astype(np.uint8)
        

def darkText(img):
    """
    Generates the textboxes candidated based on BLACKHAT morphological filter.
    Works well with dark text over bright background.
    
    Parameters
    ----------
    img : ndimage to process
    
    Returns
    -------
    mask: uint8 mask with regions of interest (possible textbox candidates)
    """
    kernel = np.ones((30, 30), np.uint8) 
    img_orig = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    
    TH = 150
    img_orig[(img_orig[:,:,0] < TH) | (img_orig[:,:,1] < TH) | (img_orig[:,:,2] < TH)] = (0,0,0)
    
    img_orig = closing(img_orig, size=(1, int(img.shape[1] / 8)))
    
    return (cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY) != 0).astype(np.uint8)


def get_textbox_score(m, p_shape):
    """
    Generates a score for how textbox-ish a mask connected component is.
    
    Parameters
    ----------
    m : mask with the textbox region with 1's
    p_shape : shape of the minimum bounding box enclosing the painting.
    
    Returns
    -------
    score: score based on size + shape
    """
    m = m.copy()
    
    # we generate the minimum bounding box for the extracted mask
    x,y,w,h = cv2.boundingRect(m.astype(np.uint8))
    
    # some upper and lower thresholding depending on its size and the painting size.
    if w < 10 or h < 10 or h > w:
        return 0
    if w >= p_shape[0]*0.8 or h >= p_shape[1]/4:
        return 0

    # we compute the score according to its shape and its size
    sc_shape = np.sum(m[y:y+h, x:x+w]) / (w*h)
    sc_size = (w*h) / (m.shape[0] * m.shape[1])
    
    final_score = (sc_shape + 50*sc_size) / 2
        
    return final_score

def get_best_textbox_candidate(mask, original_mask):
    """
    Analyzes all connected components and returns the best one according to the textbox metric.
    
    Parameters
    ----------
    m : mask with the textboxes regions with 1's
    original_mask : painting mask (size of the whole image)
    
    Returns
    -------
    score: score based on size + shape
    """
    # we will need it to crop the final textbox region so it does not goes beyond painting limits.
    x,y,w,h = cv2.boundingRect(original_mask.astype(np.uint8))
    p_shape = (w,h)
    p_coords = (x,y)
    
    # we get the biggest connected component with a score higher than TH as the textbox proposal
    mask_c = mask.copy()
    TH = 0.5
    i = 0
    found = False
    mask = None
    best_sc = 0
    while not found:
        biggest = extract_biggest_connected_component(mask_c).astype(np.uint8)
        if np.sum(biggest) == 0:
            return 0, None
        
        sc = get_textbox_score(biggest, p_shape)
        
        if sc > TH:
            mask = biggest
            best_sc = sc
            found = True
        else:
            mask_c -= biggest
            
    # we crop it and give it a margin dependant on the painting size.
    x, y, w, h = cv2.boundingRect(mask)
    M_W = 0.05
    M_H = 0.05
    ref = min(p_shape)
    x0,y0,x,y = (x - int(ref*M_W/2), y - int(ref*M_H/2), 
            (x+w) + int(ref*M_W/2), (y+h) + int(ref*M_H/2))
    return best_sc, [max(0,x0), max(0,y0), min(x, p_coords[0] + p_shape[0]), min(y, p_coords[1] + p_shape[1])]

def eval_contours(contours, width):
    if len(contours) == 0: return 0
    if len(contours) == 1: return 0
    max_area = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        max_area.append(area)
    
    max_order = [0]
    for i in range(1, len(max_area)):
        for l in range(len(max_order)+1):
            if l == len(max_order):
                max_order.append(i)
                break
            elif max_area[i] > max_area[max_order[l]]:
                max_order.insert(l, i)
                break
    # Get the moments
    mu = [None] * len(contours)
    for i in range(len(contours)):
        mu[i] = cv2.moments(contours[i])
    # Get the mass centers
    mc = [None] * len(contours)
    for i in range(len(contours)):
        # add 1e-5 to avoid division by zero
        mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i] ['m00'] + 1e-5))
    
    CM_order = [0]
    for i in range(1, len(mc)):
        for l in range(len(CM_order) + 1):
            if l == len(CM_order):
                CM_order.append(i)
                break
            elif abs(mc[i][0]-(width/2)) < abs(mc[CM_order[l]][0]-(width/2)):
                CM_order.insert(l, i)
                break
    return CM_order[0]


def estimate_noise(image) -> float:
    """
    Reference: https://stackoverflow.com/questions/2440504/noise-estimation-noise-measurement-in-image
               (J. Immerkær, "Fast Noise Variance Estimation", Computer Vision and Image Understanding)
    """
    dim = image.shape

    M = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(convolve2d(image, M))))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (dim[0]-2) * (dim[1]-2))
    return sigma
