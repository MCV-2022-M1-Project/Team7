import cv2
import numpy as np
from typing import Dict, List
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.common.registry import Registry
from src.tokenizers.base import BaseTokenizer
from typing_extensions import Protocol


def tohsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


@Registry.register_tokenizer
class VisualCodebookProcessor(BaseTokenizer):
    name: str = "visual_codebook_tokenizer"

    def __init__(self, k_size: int = 32, sample: int = 255, channel=0, num_words: int = 64, **kwargs) -> None:
        super(VisualCodebookProcessor).__init__()
        self.k_size = k_size
        self.sample = sample
        self.channel = channel
        self.num_words = num_words        

    def fit(self, images: List[np.ndarray]) -> None:
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

        for img in tqdm(images):
            img = tohsv(img)
            for i_step in range(0, img.shape[0], self.k_size):
                for j_step in range(0, img.shape[1], self.k_size):
                    hist, _ = np.histogram(
                        img[i_step:i_step+self.k_size, j_step:j_step+self.k_size, self.channel], self.sample)
                    hist = hist / hist.max()
                    self.codebook.append(hist)

        gm = KMeans(self.num_words).fit(self.codebook)
        self.bag_of_visual_words = gm

    def tokenize(self, images: List[np.ndarray]) -> Dict[str, np.ndarray]:
        features = []

        for image in images:
            image_hsv = tohsv(image)
            words_frequency_hist = np.zeros(self.num_words)

            for i_step in range(0, image_hsv.shape[0], self.k_size):
                for j_step in range(0, image_hsv.shape[1], self.k_size):
                    hist, _ = np.histogram(
                        image_hsv[i_step:i_step+self.k_size, j_step:j_step+self.k_size, self.channel], self.sample)
                    value = self.bag_of_visual_words.predict([hist])[0]
                    words_frequency_hist[value] += 1

            features.append(words_frequency_hist)
            
        return {
            "result": features
        }
