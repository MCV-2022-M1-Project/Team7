import array as np
import cv2
import numpy as np
from typing import Dict, List

from src.common.registry import Registry
from src.common.utils import TO_COLOR_SPACE, image_normalize
from src.extractors.base import FeaturesExtractor


@Registry.register_features_extractor
class GrayscaleExtractor(FeaturesExtractor):
    name: str = "grayscale_extractor"

    def __init__(self, color_space: str = "gray", channel: int = 1, *args, **kwargs) -> None:
        self.color_space: str = color_space
        self.channel: int = channel

    def run(self, images: List[np.ndarray], **kwargs) -> Dict[str, np.ndarray]:
        """
        Simple features extractor that extracts the histogram of the
        image converted to grayscale.

        Args:
            images: The list of numpy arrays representing the images.

        Returns:
            A dictionary whose result key is the list of computed histograms.
        """
        feats = []

        for image in images:
            image = cv2.resize(image, (128, 128))
            image = TO_COLOR_SPACE[self.color_space](image)

            if self.color_space != "gray":
                image = image[:,:,self.channel]

            image = image_normalize(image)
            feats.append(np.nan_to_num(image).ravel())

        return {
            "result": feats,
        }