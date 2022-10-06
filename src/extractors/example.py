import array as np
import cv2
import numpy as np
from typing import Dict, List

from src.common.utils import image_normalize
from src.common.registry import Registry
from src.extractors.base import FeaturesExtractor


@Registry.register_features_extractor
class ExampleExtractor(FeaturesExtractor):
    name: str = "example_extractor"

    def run(self, images: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Simple features extractor that extracts the histogram of the
        image converted to grayscale.

        Args:
            images: The list of numpy arrays representing the images.

        Returns:
            A dictionary whose result key is the list of computed histograms.
        """
        histograms = [image_normalize(cv2.calcHist([np.mean(img, axis=-1).astype('uint8')], [0], None, [256], [0,256])).flatten() for img in images]
        return {
            "result": histograms,
        }