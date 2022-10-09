import array as np
from threading import local
import cv2
import numpy as np
from typing import Dict, List
from scipy.stats import skew, kurtosis

from src.common.utils import image_normalize
from src.common.registry import Registry
from src.extractors.base import FeaturesExtractor


def tohsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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
        feats = [image_normalize(cv2.calcHist([np.mean(img, axis=-1).astype('uint8')], [0], None, [256], [0,256])).flatten()[1:] for img in images]
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
            bgr_planes = cv2.split(image)
            image_feats = np.concatenate([cv2.calcHist(bgr_planes, [i], None, [256], [0, 256]).ravel() for i in range(3)])
            image_feats = image_feats / np.sum(image_feats)
            image_feats_list.append(image_feats)

        return {
            "result": image_feats_list,
        }

@Registry.register_features_extractor
class HistogramMomentsExtractor(FeaturesExtractor):
    name: str = "hist_moments_extractor"

    def run(self, images: List[np.ndarray], **kwargs) -> Dict[str, np.ndarray]:
        """
        Simple features extractor that extracts the histogram of the
        image and then computes the moments.

        Args:
            images: The list of numpy arrays representing the images.

        Returns:
            A dictionary whose result key is the list of computed histograms.
        """
        image_feats_list = []

        for image in images:
            moments = []
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            for channel in range(3):
                hist, _ = np.histogram(image[:, :, channel], 255)
                hist = hist[1:]
                hist = hist/np.sum(hist)
                hist_hsv, _ = np.histogram(image_hsv[:, :, channel], 255)
                hist_hsv = hist[1:]
                hist_hsv = hist_hsv/np.sum(hist_hsv)
                moments.extend([hist_hsv.mean(), hist.mean(),
                                hist_hsv.std(), hist.std(),
                                skew(hist_hsv),  skew(hist),
                                ])
            image_feats_list.append(moments)

        return {
            "result": image_feats_list,
        }

@Registry.register_features_extractor
class HistogramThresholdExtractor(FeaturesExtractor):
    name: str = "hist_thr_extractor"

    def run(self, images: List[np.ndarray], **kwargs) -> Dict[str, np.ndarray]:
        """
        Simple features extractor that extracts the histogram of the
        image and then computes thresholded histogram.

        Args:
            images: The list of numpy arrays representing the images.

        Returns:
            A dictionary whose result key is the list of computed histograms.
        """
        image_feats_list = []

        for image in images:

            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = image_hsv[:, :, 0] <= (image_hsv[:, :, 0].mean()+image_hsv[:, :, -1].std())
            hist, _ = np.histogram(image_hsv[:, :, -1][mask], 255)
            hist = hist / (image_hsv.shape[0]*image.shape[1])
            image_feats_list.append(hist[1:])

        return {
            "result": image_feats_list,
        }

@Registry.register_features_extractor
class CumulativeHistogramExtractor(FeaturesExtractor):
    name: str = "cum_hist_extractor"

    def run(self, images: List[np.ndarray], **kwargs) -> Dict[str, np.ndarray]:
        """
        Simple features extractor that extracts the histogram of the
        image and then computes the cummulative histogram

        Args:
            images: The list of numpy arrays representing the images.

        Returns:
            A dictionary whose result key is the list of computed histograms.
        """
        image_feats_list = []

        for image in images:
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist, _ = np.histogram(image_hsv[:, :, 0], 256)
            cumhist = [sum(hist[1:i]) for i, _ in enumerate(hist)]
            image_feats_list.append(cumhist)
            
        return {
            "result": image_feats_list,
        }

@Registry.register_features_extractor
class LocalHistogramExtractor(FeaturesExtractor):
    name: str = 'local_histogram_extractor'
    def run(self, images: List[np.ndarray], n_patches: int = 10, channel: int = 0, sample: int = 255, **kwargs) -> Dict[str, np.ndarray]:

        features = []

        for image in images:
            image_hsv = tohsv(image)
            local_hists = []

            k_size_i = image_hsv.shape[0] // n_patches
            k_size_j = image_hsv.shape[1] // n_patches

            for i_step in range(0, image_hsv.shape[0] - image_hsv.shape[0]%n_patches, k_size_i):
                for j_step in range(0, image_hsv.shape[1] - image_hsv.shape[1]%n_patches, k_size_j):
                    hist, _ = np.histogram(
                        image_hsv[i_step:i_step+k_size_i, j_step:j_step+k_size_j, channel], sample)
                    local_hists.append((hist/np.sum(hist))[1:])

            image_feature = np.concatenate(local_hists)
            features.append(image_feature)
                    
        return {
            "result": features
        }

@Registry.register_features_extractor
class WeightedLocalHistogramExtractor(FeaturesExtractor):
    name: str = 'weighted_local_histogram_extractor'
    def run(self, images: List[np.ndarray], n_patches: int = 1, channel: int = 0, **kwargs) -> Dict[str, np.ndarray]:

        features = []
        sample = 256

        for image in images:
            image_hsv = tohsv(image)

            k_size_i = image_hsv.shape[0] // n_patches
            k_size_j = image_hsv.shape[1] // n_patches
            
            for i_step in range(0, image_hsv.shape[0] - image_hsv.shape[0]%n_patches, k_size_i):
                for j_step in range(0, image_hsv.shape[1] - image_hsv.shape[1]%n_patches, k_size_j):
                    hist, _ = np.histogram(
                        image_hsv[i_step:i_step+k_size_i, j_step:j_step+k_size_j, channel], sample)
                    probabilities = hist / np.sum(hist)
                    image_hsv[i_step:i_step+k_size_i, j_step:j_step+k_size_j, channel] = probabilities[image_hsv[i_step:i_step+k_size_i, j_step:j_step+k_size_j, channel]]

                            
            features.append(np.histogram(image_hsv, bins = np.linspace(0, image_hsv.max(), sample))[0])
                    
        return {
            "result": features
        }