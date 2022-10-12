import cv2
import numpy as np
from typing import Callable, Dict
from skimage import filters
from scipy.ndimage import gaussian_filter

from src.preprocessing.base import Preprocessing
from src.common.registry import Registry


def std(sample: np.ndarray) -> np.float32: return sample.std()
def var(sample: np.ndarray) -> np.float32: return sample.var()


def fill(img: np.ndarray, kernel: int = 10, iterations=1) -> np.ndarray:
    '''
    Performs filling operation by cv2 openning.

    '''
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_h, iterations=iterations)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_v, iterations=iterations)


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
            sample_image = sample_image[:, :, self.channel]

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
        # Perform AND operation in order to calculate intersection of background columns and rows.
        concat_mask = np.concatenate((horizontal_mask, vertical_mask)) 
        gaussian_filter(concat_mask, sigma=1)
        concat_thr = filters.threshold_otsu(concat_mask)
        result = (255 * ((vertical_mask > concat_thr) *
                  (horizontal_mask > concat_thr))).astype(np.uint8)

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


@Registry.register_preprocessing
class PaintThePaintingMaskPreprocessor(Preprocessing):
    name: str = "paint_mask_preprocessor"  

    def __init__(self, color_space: str = "hsv", channel: int = 0, **kwargs) -> None:
        self.color_space = TO_COLOR_SPACE[color_space]
        self.channel = channel

    def painter(self, image):
        new_image = np.zeros_like(image, dtype=np.uint8)
        opened = np.zeros_like(image)
        m, n = image.shape
        queue = [(0,0)]

        while len(queue) > 0:
            x, y = queue.pop()
            new_image[x, y] = 1
            
            if x+1 < m and image[x+1,y] == 0 and opened[x+1,y] == 0:
                opened[x+1,y] = 1
                queue.append((x+1,y))
            if x-1 >= 0 and image[x-1,y] == 0 and opened[x-1,y] == 0:
                opened[x-1,y] = 1
                queue.append((x-1,y))
            if y+1 < n and image[x,y+1] == 0 and opened[x,y+1] == 0:
                opened[x,y+1] = 1
                queue.append((x,y+1))
            if y-1 >= 0 and image[x,y-1] == 0 and opened[x,y-1] == 0:
                opened[x,y-1] = 1
                queue.append((x,y-1))

        return new_image

    def image_resize(self, image, width = None, height = None, inter = cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation = inter)

        # return the resized image
        return resized

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
        original_shape = image.shape
        # image_resize = self.image_resize(image, 400, 400)
        image_converted = self.color_space(image)
        # Select the channel we are working with from the parameter channel.
        sample_image = image_converted

        if len(image_converted.shape) > 2:
            sample_image = sample_image[:, :, self.channel]

        thr = filters.threshold_otsu(sample_image)
        mask = (sample_image > thr).astype(np.uint8)
        
        # mask = (cv2.adaptiveThreshold(sample_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        #                             cv2.THRESH_BINARY, 9, 2) == 0).astype(np.uint8)

        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 1))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 12))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel_v, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel_h, iterations=1)
        mask = self.painter(mask)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 1))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 24))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_v, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_h, iterations=1)

        mask = (mask == 0).astype(np.uint8)
        # mask = cv2.resize(mask, (original_shape[1], original_shape[0]))
        returned_image = image
        return {"result": returned_image, "mask":  mask}


@Registry.register_preprocessing
class PaintThePaintingAdaptativeMaskPreprocessor(Preprocessing):
    name: str = "painter_adaptative_mask_preprocessor"  

    def __init__(self, color_space: str = "hsv", channel: int = 0, metric: str = "std", thr_global: float = 20, fill_holes: bool = True,  **kwargs) -> None:
        self.color_space = TO_COLOR_SPACE[color_space]
        self.channel = channel
        self.metric = METRICS[metric]
        self.thr_global = thr_global
        self.fill_holes = fill_holes

    def painter(self, image):
        new_image = np.zeros_like(image, dtype=np.uint8)
        opened = np.zeros_like(image)
        m, n = image.shape
        queue = [(0,0)]

        while len(queue) > 0:
            x, y = queue.pop()
            new_image[x, y] = 1
            
            if x+1 < m and image[x+1,y] == 0 and opened[x+1,y] == 0:
                opened[x+1,y] = 1
                queue.append((x+1,y))
            if x-1 >= 0 and image[x-1,y] == 0 and opened[x-1,y] == 0:
                opened[x-1,y] = 1
                queue.append((x-1,y))
            if y+1 < n and image[x,y+1] == 0 and opened[x,y+1] == 0:
                opened[x,y+1] = 1
                queue.append((x,y+1))
            if y-1 >= 0 and image[x,y-1] == 0 and opened[x,y-1] == 0:
                opened[x,y-1] = 1
                queue.append((x,y-1))

        return new_image

    def image_resize(self, image, width = None, height = None, inter = cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation = inter)

        # return the resized image
        return resized

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
        original_shape = image.shape
        # image = self.image_resize(image, 400, 400)
        image_converted = self.color_space(image)
        # Select the channel we are working with from the parameter channel.
        sample_image = image_converted

        if len(image_converted.shape) > 2:
            sample_image = sample_image[:, :, self.channel]
            
        mask = (cv2.adaptiveThreshold(sample_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 11, 2) == 0).astype(np.uint8)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_v, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_h, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel_v, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel_h, iterations=3)
        mask = self.painter(mask)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (48, 1))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 48))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_v, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_h, iterations=1)

        mask = (mask == 0).astype(np.uint8)
        # mask = cv2.resize(mask, (original_shape[1], original_shape[0]))
        returned_image = image
        return {"result": returned_image, "mask":  mask}