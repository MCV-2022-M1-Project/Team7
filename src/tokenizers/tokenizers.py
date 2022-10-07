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
from src.tokenizers.base import Tokenizer

def tohsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

@Registry.register_tokenizer
class VisualCodebookProcessor(Tokenizer):
    name: str = "visual_codebook_tokenizer"

    def __init__(self, **kwargs) -> None:
        return None
    
    def run(self, images: List[np.array], k_size: int = 64, sample: int = 128, channel = 0, num_words: int = 64): 
        
        """
        Tokenizer that generates a visual bag of words codebook by computing local histograms on the whole dataset.
        Processor clusters the local histograms in order to create regions of feasible words.

        Args:
            images: The list of numpy arrays representing the images.
            k_size: Kernel size of the sliding window that will produce the histogram for comparing with the codebook.
            sample: Local histogram number of bins
            channel: Channel on which we compute the histograms
            num_words: Number of clusters for visual bag of words

        Returns:
            VisualCodeBookProcessor filled with the model to process bag of words
        """

        # TODO: Comments
        self.codebook: List = []
        # TODO: Compute visual codebook efficiently

        for img in images:
            img = tohsv(img)
            for i_step in range(0, img.shape[0], k_size):
                for j_step in range(0, img.shape[1], k_size):
                    hist, _ = np.histogram(img[i_step:i_step+k_size, j_step:j_step+k_size, channel], sample)
                    hist = hist / hist.max()
                    self.codebook.append(hist)

        gm = KMeans(num_words).fit(self.codebook)

        self.k_size = k_size
        self.sample = sample
        self.channel = channel
        self.num_words = num_words
        self.bag_of_visual_words = gm

        return self
    
    def fit(self, images: List[np.array], k_size: int = 64, sample: int = 128, channel = 0, num_words: int = 64):
        return self.run(images, k_size, sample, channel, num_words)
    
    def tokenize(self, sample: np.ndarray):

        features = []
        img = tohsv(sample)
        words_frequency_hist = np.zeros(self.num_words)
        for i_step in range(0, img.shape[0], self.k_size):
            for j_step in range(0, img.shape[1], self.k_size):
                hist, _ = np.histogram(img[i_step:i_step+self.k_size, j_step:j_step+self.k_size, self.k_size], sample)
                value = self.bag_of_visual_words.predict([hist])[0]
                words_frequency_hist[value] += 1
        features.append(words_frequency_hist)

        return {
            "result": features 
        }
            