import cv2
import numpy as np
from typing import Callable, Dict, Tuple

from skimage.restoration import denoise_wavelet, estimate_sigma

from src.common.utils import estimate_noise
from src.preprocessing.base import Preprocessing
from src.common.registry import Registry


@Registry.register_preprocessing
class ShadowRemovePreprocessor(Preprocessing):
    name: str = "shadow_remove_preprocessor"

    def run(self, image, **kwargs) -> Dict[str, np.ndarray]:
        """

        Takes an image as an input and applies morphological transformation to remove shadows.
        Reference: https://stackoverflow.com/questions/44752240/how-to-remove-shadow-from-scanned-images-using-opencv

        Args:
            image: Sample image to preprocess

        Returns:
            Dict: {
                "output": Processed image.
            }

        """
        rgb_planes = cv2.split(image)
        result_norm_planes = []

        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_norm_planes.append(norm_img)

        enhanced = cv2.merge(result_norm_planes)
        return {"result": enhanced}


@Registry.register_preprocessing
class ChannelsDenoisePreprocessor(Preprocessing):
    name: str = "channels_denoise_preprocessor"

    def __init__(self, h=3, template_window_size=7, search_window_size=21, **kwargs) -> None:
        self.h = h
        self.template_window_size = template_window_size
        self.search_window_size = search_window_size

    def run(self, image, **kwargs) -> Dict[str, np.ndarray]:
        """

        Takes an image as an input and applies morphological transformations to remove noise.

        Args:
            image: Sample image to preprocess

        Returns:
            Dict: {
                "output": Processed image.
            }

        """
        enhanced = image
        if estimate_noise(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) > 10:
            (b, g, r) = cv2.split(image)

            b_blur = cv2.medianBlur(b, 3)
            g_blur = cv2.medianBlur(g, 3)
            r_blur = cv2.medianBlur(r, 3)

            b_denoise = cv2.fastNlMeansDenoising(b_blur, self.h, self.template_window_size, self.search_window_size)
            g_denoise = cv2.fastNlMeansDenoising(g_blur, self.h, self.template_window_size, self.search_window_size)
            r_denoise = cv2.fastNlMeansDenoising(r_blur, self.h, self.template_window_size, self.search_window_size)

            enhanced = cv2.merge((b_denoise, g_denoise, r_denoise))

        return {"result": enhanced}


@Registry.register_preprocessing
class ColoredDenoisePreprocessor(Preprocessing):
    name: str = "colored_denoise_preprocessor"

    def __init__(self, h=10, template_window_size=7, search_window_size=21, **kwargs) -> None:
        self.h = h
        self.template_window_size = template_window_size
        self.search_window_size = search_window_size

    def run(self, image, **kwargs) -> Dict[str, np.ndarray]:
        """

        Takes an image as an input and applies morphological transformations to remove noise.

        Args:
            image: Sample image to preprocess

        Returns:
            Dict: {
                "output": Processed image.
            }

        """
        enhanced = image
        if estimate_noise(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) > 10:
            blur = cv2.medianBlur(image, 3)
            enhanced = cv2.fastNlMeansDenoisingColored(blur, self.h, self.template_window_size, self.search_window_size)

        return {"result": enhanced}


@Registry.register_preprocessing
class VisuShrinkDenoisePreprocessor(Preprocessing):
    name: str = "visushrink_noise_preprocessor"

    def __init__(self, convert2ycbcr=True, mode='soft', rescale_sigma=True, **kwargs) -> None:
        self.convert2ycbcr = convert2ycbcr
        self.mode = mode
        self.rescale_sigma = rescale_sigma

    def run(self, image, **kwargs) -> Dict[str, np.ndarray]:
        """

        Takes an image as an input and applies morphological transformations to remove noise.
        VisuShrink is designed to eliminate noise with high probability, but this results in an over-smooth appearance.
        Args:
            image: Sample image to preprocess

        Returns:
            Dict: {
                "output": Processed image.
            }

        """
        enhanced = image
        if estimate_noise(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) > 10:
            blur = cv2.medianBlur(image, 3)
            sigma_est = estimate_sigma(blur, channel_axis=-1, average_sigmas=True)
            enhanced = denoise_wavelet(blur, channel_axis=-1, convert2ycbcr=self.convert2ycbcr,
                                       method='VisuShrink', mode=self.mode,
                                       sigma=sigma_est/2, rescale_sigma=self.rescale_sigma)
            enhanced = enhanced*255
        return {"result": enhanced.astype(np.uint8)}
