import array as np
from threading import local
import cv2
import numpy as np
import scipy.stats as stats
from typing import Dict, List
from scipy import spatial
from sklearn.metrics import euclidean_distances
from sympy import euler
from sklearn.cluster import KMeans

from src.common.utils import image_normalize
from src.common.registry import Registry
from src.extractors.base import FeaturesExtractor


# Vsual codebook
def get_artificial_codebook(sample_hz: int = 255, max_exp: int = 25, norm_max: int = 1):

    ### WHAT IF CODEBOOKS ARE RANDOM NOISE COMPOSITION OF SIGNALS????
    
    norm_signal = lambda x: norm_max * ((x - x.min())/(x.max() - x.min()))
    
    ### Normal
    mu = 0
    variance = 1
    sigma = variance**.5
    
    xnormal = np.linspace(mu - 3*sigma, mu + 3*sigma, sample_hz)
    y_normal = stats.norm.pdf(xnormal, mu, sigma)    
    
    ### Normal Inverse
    y_normal_inverse = -y_normal
    
    
    ### Exponential - Left
    y_exp = np.linspace(0, max_exp, sample_hz)
    y_exp_left = stats.expon.pdf(y_exp)

    
    
    ### Exponential Inverse - Left
    y_exp_inv_left = - y_exp_left
    
    
    ### Exponential - Right
    y_exp_right = y_exp_inv_left[::-1]
    
    
    ### Exponential Inverse - Right
    y_exp_inv_right = - y_exp_right
    
    
    
    ### Constant
    y_constant = np.ones(sample_hz)
    
    
    ### Normalize Distributions y: (0-1)
    distributions = [y_normal, y_normal_inverse,
                    y_exp_left, y_exp_inv_left,
                    y_exp_right, y_exp_inv_right,]
    return [norm_signal(x) for x in distributions] + [y_constant]

def tohsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

@Registry.register_features_extractor
class VisualCodebookExtractor(FeaturesExtractor):
    name: str = "visual_codebook_extractor"

    def run(self, images: List[np.ndarray], codebook: object = None) -> Dict[str, np.ndarray]:

        """
        Extractor that process the histogram of occurences of a certain visual word with respect a fixed codebook.
        The histogram of codebook appearences acts as feature for the image. Given a certain path, its codebook is the one with maximum cosine similarity.
        Minimum similarity is required in order to belong to a codebook.


        Args:
            images: The list of numpy arrays representing the images.
            k_size: Kernel size of the sliding window that will produce the histogram for comparing with the codebook.

        Returns:
            A dictionary whose result key is the list of computed histograms.
        """
        assert isinstance(codebook), f'Visual Codebook Extractor must receive a codebook as a parameter. Received: {codebook}'

        k_size = codebook.k_size
        channel = codebook.channel
        sample = codebook.channel

        features = []
        for img in images:
            img = tohsv(img)
            words_frequency_hist = np.zeros(codebook.num_words)
            for i_step in range(0, img.shape[0], k_size):
                for j_step in range(0, img.shape[1], k_size):
                    hist, _ = np.histogram(img[i_step:i_step+k_size, j_step:j_step+k_size, channel], sample)
                    value = codebook.bag_of_visual_words.predict([hist])[0]
                    words_frequency_hist[value] += 1
            features.append(words_frequency_hist)

        return {
            "result": features 
        }

@Registry.register_features_extractor
class RandomFeaturesExtractor(FeaturesExtractor):
    name: str = "random_features_extractor"

    def run(self, images: List[np.ndarray],) -> Dict[str, np.ndarray]:
        """
        Gets a bunch of random list of numbers so we have a baseline on defininng where's the minimum of the function.
        # TODO: Explain this better


        Args:
            images: The list of numpy arrays representing the images.

        Returns:
            A dictionary whose result key is the list of computed histograms.
        """



        return {
            "result": np.random.rand(len(images), 10),
        }