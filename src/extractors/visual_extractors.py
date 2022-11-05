import array as np
import cv2
import numpy as np
import scipy.stats as stats
import concurrent
from typing import Dict, List

from src.common.registry import Registry
from src.extractors.base import FeaturesExtractor
from src.tokenizers.base import BaseTokenizer


# Vsual codebook
def get_artificial_codebook(sample_hz: int = 255, max_exp: int = 25, norm_max: int = 1):

    # WHAT IF CODEBOOKS ARE RANDOM NOISE COMPOSITION OF SIGNALS????

    def norm_signal(x): return norm_max * ((x - x.min())/(x.max() - x.min()))

    # Normal
    mu = 0
    variance = 1
    sigma = variance**.5

    xnormal = np.linspace(mu - 3*sigma, mu + 3*sigma, sample_hz)
    y_normal = stats.norm.pdf(xnormal, mu, sigma)

    # Normal Inverse
    y_normal_inverse = -y_normal

    ### Exponential - Left
    y_exp = np.linspace(0, max_exp, sample_hz)
    y_exp_left = stats.expon.pdf(y_exp)

    # Exponential Inverse - Left
    y_exp_inv_left = - y_exp_left

    ### Exponential - Right
    y_exp_right = y_exp_inv_left[::-1]

    # Exponential Inverse - Right
    y_exp_inv_right = - y_exp_right

    # Constant
    y_constant = np.ones(sample_hz)

    # Normalize Distributions y: (0-1)
    distributions = [y_normal, y_normal_inverse,
                     y_exp_left, y_exp_inv_left,
                     y_exp_right, y_exp_inv_right, ]
    return [norm_signal(x) for x in distributions] + [y_constant]


def tohsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


@Registry.register_features_extractor
class VisualCodebookExtractor(FeaturesExtractor):
    name: str = "visual_codebook_extractor"

    def run(self, images: List[np.ndarray], tokenizer: BaseTokenizer = None) -> Dict[str, np.ndarray]:
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
        assert isinstance(
            tokenizer, BaseTokenizer), f'Visual Codebook Extractor must receive a tokenizer as a parameter. Received: {tokenizer}'
        features = tokenizer.tokenize(images)["result"]
        return {
            "result": features
        }


@Registry.register_features_extractor
class RandomFeaturesExtractor(FeaturesExtractor):
    name: str = "random_features_extractor"

    def run(self, images: List[np.ndarray], **kwargs) -> Dict[str, np.ndarray]:
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


@Registry.register_features_extractor
class SIFTExtractor(FeaturesExtractor):
    name: str = "sift_features_extractor"

    def __init__(self, n_keypoints=40, n_threads=4, *args, **kwargs) -> None:
        self.n_keypoints = n_keypoints
        self.sift_size = 128
        self.descriptor_size = n_keypoints * self.sift_size
        self.n_threads = n_threads

    def process_imgs_mp(self, data):
        image = data[0]
        sift = cv2.SIFT_create()

        keypoints, descriptors = sift.detectAndCompute(image, None)

        if not isinstance(descriptors, np.ndarray):
            descriptors = np.zeros((1, self.sift_size), dtype=np.float32)

        return descriptors

        # if len(keypoints) > self.n_keypoints:
        #     descriptors = descriptors[:self.n_keypoints, :]
        # left = self.n_keypoints if not isinstance(
        #     descriptors, np.ndarray) else self.n_keypoints - len(descriptors)

        # if left:
        #     fill = np.zeros(left * self.sift_size, dtype=np.float32)
        #     if isinstance(descriptors, np.ndarray):
        #         fill = np.hstack((descriptors.reshape(-1), fill))
        #     return fill
        # else:
        #     return descriptors.reshape(-1).astype(np.float32)
                
        # if len(features[n]) != self.descriptor_size:
        #     print(descriptors)

    def run(self, images: List[np.ndarray], **kwargs) -> Dict[str, np.ndarray]:
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_threads) as executor:
            features = list(executor.map(
                self.process_imgs_mp, 
                [(img, ) for img in images]
                ))

        # mp_manager = mp.Manager()
        # features = mp_manager.list([None for i in range(len(images))])
        
        # threads = [mp.Process(target=self.process_imgs_mp, args=(
        #     images, features, i)) for i in range(self.n_threads)]
        # [t.start() for t in threads]
        # [t.join() for t in threads]

        return {
            "result": features,
        }
@Registry.register_features_extractor
class ORBExtractor(FeaturesExtractor):
    name: str = "orb_features_extractor"
    def __init__(self, *args, **kwargs)->None:
        pass
    def run(self, images: List[np.ndarray], **kwargs) -> Dict[str, np.ndarray]:
        result = []
        orb = cv2.ORB_create()
        for image in images:
            kp, des = orb.detectAndCompute(image,None)
            #print(kp,des)
            if(des  is not None):
                result.append(des.reshape(-1))
                print(len(des))
        return {"result": result}
