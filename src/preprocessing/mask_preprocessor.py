import cv2
import numpy as np
from typing import Callable, Dict

from src.preprocessing.base import Preprocessing
from src.common.registry import Registry


def std(sample: np.ndarray) -> np.float32: return sample.std()
def var(sample: np.ndarray) -> np.float32: return sample.var()


def fill(img: np.ndarray, kernel: tuple = (10, 10), iterations=1) -> np.ndarray:
    '''
    Performs filling operation by cv2 openning.

    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def tohsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def tolab(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


def togray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


TO_COLOR_SPACE: Dict[str, Callable] = {
    "hsv": tohsv,
    "lab": tolab,
    "gray": togray,
}


METRICS: Dict[str, Callable] = {
    "std": std,
    "var": var
}


@Registry.register_preprocessing
class VarianceMaskPreprocessor(Preprocessing):
    name: str = "variance_mask_preprocessor"

    def __init__(self, color_space: str = "hsv", channel: int = 0, metric: str = "std", thr_global: float = 20, fill_holes: bool = True,  **kwargs) -> None:
        self.color_space = TO_COLOR_SPACE[color_space]
        self.channel = channel
        self.metric = METRICS[metric]
        self.thr_global = thr_global
        self.fill_holes = fill_holes

    def run(self,  image, **kwargs) -> Dict[str, np.ndarray]:
        '''

        run(**kwargs) takes an image as an imput and process standard deviation of each row and column.
        Thresholds and operate boolean and for masking.

        Some asumptions made here: 
            1. Background is on the sides of the image.
            2. Painting is on the center of the image.
            3. Background is the least entropic region (lower variance) of the image. In other words: Walls are more boring than paintings.
            4. Low-entropy in background produces "spike" on histogram, which is characterized by lower variance.
            5. Photo of the painting isn't tilted. Thus, we can scan it iteratively.

        Args:

            Image: Sample image to preprocess
            Channel: Channel we are scanning
            Metric: Metric used to calculate the least entropic channel, variance has more predictable behaviour.
            thr_global: Threshold of minimum variance to be considered as possitive sample.
            fill_holes: Boolean. Some cases, when painting is composed by sub-paintings it detects the sub-painting level. 
                        We can solve this in order to adjust to the GT by filling the holes.

        Returns:
            Dict: {
                "ouput": Processed image cropped with mask
                "mask": mask obtained with method 
            }

        '''

        # TODO: Precondition: Channel first or channel last?
        image_converted = self.color_space(image)
        # Select the channel we are working with from the parameter channel.
        sample_image = image_converted

        if len(image_converted.shape) > 2:
            sample_image = sample_image[:, :, self.channel] * 1.5

        horizontal_mask = np.zeros_like(
            sample_image)  # Create masks for scanning
        vertical_mask = np.zeros_like(horizontal_mask)
        shape = image.shape

        ### Vertical scan ###
        for col in range(shape[0]):

            row_vector = sample_image[col, :]  # Extract a particular column
            # Set the mask to its metric level
            vertical_mask[col, :] = self.metric(row_vector)

        ### Horizontal scan ###
        for row in range(shape[1]):
            row_vector = sample_image[:, row]
            horizontal_mask[:, row] = self.metric(row_vector)

        # Perform thresholding to minimum variance required and type to cv2 needs.
        result = (255 * ((vertical_mask > self.thr_global) *
                  (horizontal_mask > self.thr_global))).astype(np.uint8)
        # Perform AND operation in order to calculate intersection of background columns and rows.
        if self.fill_holes:
            # If fill-holes is set to true, fill the image holes.
            result = fill(result)
        result *= 1
        returned_image = image.copy()
        for i in range(3):

            returned_image[:, :, i][result] = 0

        return {"result": returned_image, "mask":  (result != 0).astype(np.uint8)}


@Registry.register_preprocessing
class LocalVarianceMaskPreprocessor(Preprocessing):
    name: str = "local_variance_mask_preprocessor"

    def __init__(self,  channel: int = 0, kernel_size: int = 10, thr_global: float = 2.5,  **kwargs) -> None:
        self.channel = channel
        self.kernel_size = kernel_size
        self.thr_global = thr_global

    def run(self, image: np.ndarray,  **kwargs) -> Dict[str, np.ndarray]:
        '''

        run(**kwargs) takes an image as an imput and process standard deviation of local areas of fixed size.
        Thresholds and operate boolean and for masking.

        Some asumptions made here: 
            1. Background is the least entropic region (lower variance) of the image. In other words: Walls are more boring than paintings.
            2. Low-entropy in background produces "spike" on histogram, which is characterized by lower variance.

        Args:

            image: Sample image to preprocess
            channel: Channel we are scanning
            thr_global: Threshold of minimum variance to be considered as possitive sample.
            kernel_size: Size of the sliding window we are using in order to compute local histograms


        Returns:
            Dict: {
                "ouput": Processed image cropped with mask
                "mask": mask obtained with method 
            }

        '''

        # TODO: Precondition: Channel first or channel last? + Comments
        image_hsv = tohsv(image)
        # Select the channel we are working with from the parameter channel.
        sample_image = image_hsv[:, :, self.channel]
        mask = np.zeros_like(sample_image)
        shape = image.shape

        for step_i in range(0, shape[0], self.kernel_size):
            for step_j in range(0, shape[1], self.kernel_size):

                patch = sample_image[step_i:step_i +
                                     self.kernel_size, step_j:step_j+self.kernel_size]
                mask[step_i:step_i+self.kernel_size,
                     step_j:step_j+self.kernel_size] = patch.std()

        mask = mask > self.thr_global
        returned_image = image.copy()
        for i in range(3):

            returned_image[:, :, i][mask] = 0
        return {"result": returned_image, "mask":  (mask != 0).astype(np.uint8)}


@Registry.register_preprocessing
class CombinedMaskPreprocessor(Preprocessing):
    name: str = "combined_mask_preprocessor"

    def __init__(self, channel: int = 0, kernel_size: int = 5, thr_global: float = 20, thr_local: float = 5, fill_holes: bool = True, metric: Callable = std, **kwarg) -> None:
        self.channel = channel
        self.kernel_size = kernel_size
        self.thr_global = thr_global
        self.thr_local = thr_local
        self.fill_holes = fill_holes
        self.metric = metric

    def run(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        '''

        run(**kwargs) takes an image as an imput and process both LocalVarianceMaskPreprocessor and VarianceMaskPreprocessor.
        This cleans the output on the first processer allowing the second one to process tilted images.

        Args:

            image: Sample image to preprocess
            channel: Channel we are scanning
            thr_global: Threshold of minimum variance to be considered as possitive sample.
            kernel_size: Size of the sliding window we are using in order to compute local histograms


        Returns:
            Dict: {
                "ouput": Processed image cropped with mask
                "mask": mask obtained with method 
            }

        '''

        res_global = VarianceMaskPreprocessor(
            self.channel, self.metric, self.thr_global, self.fill_holes).run(image)['mask'] != 0
        res_local = LocalVarianceMaskPreprocessor(
            self.channel, self.kernel_size, self.thr_local).run(image)['mask'] != 0
        mask = (255 * res_global * res_local).astype(np.uint8)
        if self.fill_holes:
            fill(mask, iterations=7)
        mask *= 1
        returned_image = image.copy()
        for i in range(3):

            returned_image[:, :, i][mask] = 0
        return {"result": returned_image, "mask": (mask != 0).astype(np.uint8)}


@Registry.register_preprocessing
class MeanVarianceMaskPreprocessor(Preprocessing):
    name: str = "meanvariance_mask_preprocessor"

    def __init__(self, color_space: str = "hsv", channel: int = 0, metric: str = "std", thr_global: float = 20, fill_holes: bool = True,  **kwargs) -> None:
        self.color_space = TO_COLOR_SPACE[color_space]
        self.channel = channel
        self.metric = METRICS[metric]
        self.thr_global = thr_global
        self.fill_holes = fill_holes

    def run(self,  image, **kwargs) -> Dict[str, np.ndarray]:
        '''

        run(**kwargs) takes an image as an imput and process standard deviation of each row and column.
        Thresholds and operate boolean and for masking.

        Some asumptions made here: 
            1. Background is on the sides of the image.
            2. Painting is on the center of the image.
            3. Background is the least entropic region (lower variance) of the image. In other words: Walls are more boring than paintings.
            4. Low-entropy in background produces "spike" on histogram, which is characterized by lower variance.
            5. Photo of the painting isn't tilted. Thus, we can scan it iteratively.

        Args:

            Image: Sample image to preprocess
            Channel: Channel we are scanning
            Metric: Metric used to calculate the least entropic channel, variance has more predictable behaviour.
            thr_global: Threshold of minimum variance to be considered as possitive sample.
            fill_holes: Boolean. Some cases, when painting is composed by sub-paintings it detects the sub-painting level. 
                        We can solve this in order to adjust to the GT by filling the holes.

        Returns:
            Dict: {
                "ouput": Processed image cropped with mask
                "mask": mask obtained with method 
            }

        '''

        # TODO: Precondition: Channel first or channel last?
        image_converted = self.color_space(image)
        # Select the channel we are working with from the parameter channel.
        sample_image = image_converted

        if len(image_converted.shape) > 2:
            sample_image = sample_image[:, :, self.channel]

        horizontal_mask = np.zeros_like(
            sample_image)  # Create masks for scanning
        vertical_mask = np.zeros_like(horizontal_mask)
        shape = image.shape

        ### Vertical scan ###
        for col in range(shape[0]):
            col_vector = sample_image[col, :]
            col_std = std(col_vector)
            col_mean = np.mean(col_vector)
            vertical_mask[col, :] = np.abs(col_vector - col_mean) > col_std

        ### Horizontal scan ###
        for row in range(shape[1]):
            row_vector = sample_image[:, row]
            row_std = std(row_vector)
            row_mean = np.mean(row_vector)
            horizontal_mask[:, row] = np.abs(row_vector - row_mean) > row_std

        # Perform thresholding to minimum variance required and type to cv2 needs.
        result = (255 * (vertical_mask * horizontal_mask)).astype(np.uint8)
        # Perform AND operation in order to calculate intersection of background columns and rows.
        if self.fill_holes:
            # If fill-holes is set to true, fill the image holes.
            result = fill(result)
        result *= 1
        returned_image = image.copy()
        for i in range(3):

            returned_image[:, :, i][result] = 0

        return {"result": returned_image, "mask":  (result != 0).astype(np.uint8)}


@Registry.register_preprocessing
class MeanVarianceIterMaskPreprocessor(Preprocessing):
    name: str = "meanvarianceiter_mask_preprocessor"

    def __init__(self, color_space: str = "hsv", channel: int = 0, metric: str = "std", thr_global: float = 20, fill_holes: bool = True,  **kwargs) -> None:
        self.color_space = TO_COLOR_SPACE[color_space]
        self.channel = channel
        self.metric = METRICS[metric]
        self.thr_global = thr_global
        self.fill_holes = fill_holes

    def run(self,  image, **kwargs) -> Dict[str, np.ndarray]:
        '''

        run(**kwargs) takes an image as an imput and process standard deviation of each row and column.
        Thresholds and operate boolean and for masking.

        Some asumptions made here: 
            1. Background is on the sides of the image.
            2. Painting is on the center of the image.
            3. Background is the least entropic region (lower variance) of the image. In other words: Walls are more boring than paintings.
            4. Low-entropy in background produces "spike" on histogram, which is characterized by lower variance.
            5. Photo of the painting isn't tilted. Thus, we can scan it iteratively.

        Args:

            Image: Sample image to preprocess
            Channel: Channel we are scanning
            Metric: Metric used to calculate the least entropic channel, variance has more predictable behaviour.
            thr_global: Threshold of minimum variance to be considered as possitive sample.
            fill_holes: Boolean. Some cases, when painting is composed by sub-paintings it detects the sub-painting level. 
                        We can solve this in order to adjust to the GT by filling the holes.

        Returns:
            Dict: {
                "ouput": Processed image cropped with mask
                "mask": mask obtained with method 
            }

        '''

        # TODO: Precondition: Channel first or channel last?
        image_converted = self.color_space(image)
        # Select the channel we are working with from the parameter channel.
        sample_image = image_converted

        if len(image_converted.shape) > 2:
            sample_image = sample_image[:, :, self.channel]

        # Create masks for scanning
        horizontal_mask = np.zeros(sample_image.shape)
        vertical_mask = np.zeros_like(horizontal_mask)
        shape = image.shape

        background_means = (np.mean(sample_image[0, :])
                           + np.mean(sample_image[-1, :])
                           + np.mean(sample_image[:, 0])
                           + np.mean(sample_image[:, -1])) / 4
        sample_stds = np.std(sample_image)

        ### Vertical scan ###
        for col in range(shape[0]):
            col_vector = sample_image[col, :]
            # col_std = std(col_vector)
            # col_mean = np.mean(col_vector)
            inside_frame = False

            for i in range(shape[1]):
                val = col_vector[i]

                if val - background_means > 0:
                    inside_frame = True
                else:
                    inside_frame = False

                vertical_mask[col, i] = 1 if inside_frame else 0

        ### Horizontal scan ###
        for row in range(shape[1]):
            row_vector = sample_image[:, row]
            # row_std = std(row_vector)
            # row_mean = np.mean(row_vector)
            inside_frame = False

            for i in range(shape[0]):
                val = row_vector[i]

                if val - background_means > 0:
                    inside_frame = True
                else:
                    inside_frame = False

                horizontal_mask[i, row] = 1 if inside_frame else 0

        # Perform thresholding to minimum variance required and type to cv2 needs.
        result = (255 * (vertical_mask * horizontal_mask)).astype(np.uint8)
        # Perform AND operation in order to calculate intersection of background columns and rows.
        if self.fill_holes:
            # If fill-holes is set to true, fill the image holes.
            result = fill(result)

        result *= 1
        returned_image = image.copy()
        for i in range(3):

            returned_image[:, :, i][result] = 0

        return {"result": returned_image, "mask":  (result != 0).astype(np.uint8)}
