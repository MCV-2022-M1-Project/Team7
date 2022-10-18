import cv2
import numpy as np
from typing import Callable, Dict, Tuple
from skimage import filters
from scipy.ndimage import gaussian_filter

from src.preprocessing.base import Preprocessing
from src.common.registry import Registry


@Registry.register_preprocessing
class ShadowRemovePreprocessor(Preprocessing):
    name: str = "shadow_remove_preprocessor"

    def run(self,  image, **kwargs) -> Dict[str, np.ndarray]:
        '''

        Takes an image as an imput and applies momrphological transformation to remove shadows.
        Reference: https://stackoverflow.com/questions/44752240/how-to-remove-shadow-from-scanned-images-using-opencv

        Args:
            Image: Sample image to preprocess

        Returns:
            Dict: {
                "ouput": Processed image.
            }

        '''
        rgb_planes = cv2.split(image)
        result_norm_planes = []

        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_norm_planes.append(norm_img)

        enhanced = cv2.merge(result_norm_planes)
        return {"result": enhanced}