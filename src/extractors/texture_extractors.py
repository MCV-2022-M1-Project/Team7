from typing import Dict, List
import numpy as np
import cv2
import mahotas
#import os
#from functools import partial
from skimage.feature import local_binary_pattern
#import multiprocessing.dummy as mp
import pywt

from src.common.registry import Registry
from src.extractors.base import FeaturesExtractor

@Registry.register_features_extractor
class HistogramOrientedGradientsExtractor(FeaturesExtractor):
    name: str = 'HOG_extractor'

    def __init__(self, win_size = 16, block_size = 16, block_stride = 8, cell_size = 8, n_bins = 180, img_shape = 240, *args, **kwargs) -> None:

        self.hog_descriptor = cv2.HOGDescriptor((win_size, win_size), (block_size, block_size), (block_stride, block_stride), (cell_size, cell_size), n_bins)
        self.shape = (img_shape, img_shape)


        return None
    
    def run(self, images: List[np.ndarray], **kwargs) -> Dict[str, np.ndarray]:

        return {
            'result': [self.hog_descriptor.compute(cv2.resize(image, self.shape)) for image in images]
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
        
@Registry.register_features_extractor
class LocalBinaryPatternExtractor(FeaturesExtractor):
    name: str = 'lbp_extractor'
    def __init__(self, points:int=24, radius:float=3.0, bins:int=8, mask:np.ndarray=None, *args, **kwargs) -> None:
        self.radius = radius
        self.points = points
        self.bins = bins
        self.mask = mask
    
    def run(self, images:List[np.array], **Kwargs) -> Dict[str, np.ndarray]:
        """
        Extract LBP descriptors after dividing image in non-overlapping blocks,
        computing histograms for each block and then concatenating them
        
        Args:
            image: (H x W x C) 3D BGR image array of type np.uint8
            points: number of circularly symmetric neighbour set points (quantization of the angular space)
            radius: radius of circle (spatial resolution of the operator)
            bins: number of bins to use for histogram
            mask: check _descriptor(first function in file)
        Returns:
            Histogram features flattened into a 
            1D array of type np.float32
        """
        result = []
        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = (local_binary_pattern(image, self.points, self.radius, method="uniform")).astype(np.uint8)
            self.bins = self.points + 2
            hist = cv2.calcHist([image],[0], self.mask, [self.bins], [0, self.bins])
            hist = cv2.normalize(hist, hist)
            result.append(hist.flatten())
        return {'result': result}
    
@Registry.register_features_extractor
class DiscreteCosineTransformExtractor(FeaturesExtractor):
    name: str = 'dct_extractor'
    def __init__(self, bins:int=8, mask:np.ndarray=None, num_coeff:int=4, *args, **kwargs) -> None:
        self.bins = bins
        self.mask = mask
        self.num_coeff = num_coeff
        
    def run(self, images:List[np.array], **Kwargs) -> Dict[str, np.ndarray]:
        """
        Extract DCT coefficients from image. This descriptor will be clubbed with a block descriptor
        
        Args:
            image: (H x W x C) 3D BGR image array of type np.uint8
            num_coeff: number of coefficents in dct_block to use through zig-zag scan
            bins: N.A. here, but present to make our api compatible with the function
            mask: check _descriptor(first function in file)
        Returns:
            DCT features flattened into a 
            1D array of type np.float32
        """
        def _compute_zig_zag(a):
            return np.concatenate([np.diagonal(a[::-1,:], k)[::(2*(k % 2)-1)] for k in range(1-a.shape[0], a.shape[0])])
        result = []
        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            if self.mask is not None:
                image = cv2.bitwise_and(image, image, mask=self.mask)
            
            block_dct = cv2.dct(np.float32(image)/255.0)
            
            features = _compute_zig_zag(block_dct[:6,:6])[:self.num_coeff]
            
            result.append(features)
            
        return {'result': result}
    
@Registry.register_features_extractor
class DiscreteWaveletTransformExtractor(FeaturesExtractor):
    name: str = 'dwt_extractor'
    def __init__(self, level: int =3, n_coefs: int =7, wavelet: str ='db1', *args, **Kwargs) -> None:
            self.level = level
            self.n_coefs = n_coefs
            self.wavelet = wavelet
    
    def run(self, images:List[np.array], **Kwargs) -> Dict[str, np.ndarray]:
        result = []
        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            extracted_coefs = pywt.wavedec2(gray, wavelet=self.wavelet, level=self.level)
            first_coef, *level_coefs = extracted_coefs
            wavelet_coefs = []
            wavelet_coefs.append(first_coef)
            
            for i in range(self.level):
                (LH, HL, HH) = level_coefs[i]
                wavelet_coefs.append(LH)
                wavelet_coefs.append(HL)
                wavelet_coefs.append(HH)
                
            wavelet_coefs = wavelet_coefs[:self.n_coefs]
            hist_concat = None
            
            for cf in wavelet_coefs:
                max_range = abs(np.amax(cf))+1
                hist = cv2.calcHist([cf.astype(np.uint8)], [0], None, [8], [0, max_range])
                
                if hist_concat is None:
                    hist_concat = hist
                else:
                    hist_concat = cv2.hconcat([hist_concat, hist])
                    
            result.append(hist_concat.flatten())
        return {'result': result}
