import array as np
import cv2
import numpy as np
from typing import Dict, List

from src.common.utils import image_normalize
from src.common.registry import Registry
from src.extractors.base import FeaturesExtractor


@Registry.register_features_extractor
class HistogramGrayscaleExtractor(FeaturesExtractor):
    name: str = "hist_grayscale_extractor"

    def run(self, images: List[np.ndarray], **kwargs) -> Dict[str, np.ndarray]:
        """
        Simple features extractor that extracts the histogram of the
        image converted to grayscale.

        Args:
            images: The list of numpy arrays representing the images.

        Returns:
            A dictionary whose result key is the list of computed histograms.
        """
        feats = [image_normalize(cv2.calcHist([np.mean(img, axis=-1).astype('uint8')], [0], None, [256], [0,256])).flatten() for img in images]
        return {
            "result": feats,
        }


@Registry.register_features_extractor
class HistogramRGBConcatExtractor(FeaturesExtractor):
    name: str = "hist_rgb_concat_extractor"

    def run(self, images: List[np.ndarray], **kwargs) -> Dict[str, np.ndarray]:
        """
        Simple features extractor that extracts the histogram of the
        image converted to grayscale.

        Args:
            images: The list of numpy arrays representing the images.

        Returns:
            A dictionary whose result key is the list of computed histograms.
        """
        image_feats_list = []

        for image in images:
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            bgr_planes = cv2.split(image_hsv)
            # image_feats = np.concatenate([cv2.calcHist(bgr_planes, [i], None, [256], [0, 256]).ravel() for i in range(3)])
            image_feats = cv2.calcHist(bgr_planes, [0], None, [256], [0, 256]).ravel() 
            image_feats = image_feats / np.max(image_feats)
            image_feats_list.append(image_feats)

        return {
            "result": image_feats_list,
        }