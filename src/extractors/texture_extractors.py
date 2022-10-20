from this import d
from typing import Dict, List
import numpy as np
import cv2
import mahotas

from src.common.registry import Registry
from src.extractors.base import FeaturesExtractor

@Registry.register_features_extractor
class HistogramOrientedGradientsExtractor(FeaturesExtractor):
    name: str = 'HOG_extractor'

    def __init__(self, win_size = 64, block_size = 16, block_stride = 8, cell_size = 8, n_bins = 180, *args, **kwargs) -> None:

        self.hog_descriptor = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins)

        return None
    
    def run(self, images: List[np.ndarray], **kwargs) -> Dict[str, np.ndarray]:
        return {
            'result': [self.hog_descriptor.compute(image) for image in images]
        }

@Registry.register_features_extractor
class ZernikeExtractor(FeaturesExtractor):
    name: str = 'zernike_extractor'
    def __init__(self, radius: int = 10, degree = 12, channel: int = 1, *args, **kwargs) -> None:
        self.radius = radius
        self.degree = degree
        self.channel = channel
    
    def run(self, images: List[np.array], **kwargs) -> Dict:
        return {
            'result': [mahotas.features.zernike_moments(image[:, :, self.channel], self.radius, self.degree) for image in images]
        }