import logging

import cv2
import numpy as np
from typing import Callable, Dict
from skimage import filters
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance as dist

from src.common.utils import TO_COLOR_SPACE, image_resize
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

    def run(self,  image, **kwargs) -> Dict[str, np.ndarray]:
        '''

        run(**kwargs) takes an image as an imput and applies morhpology operators to
        create the mask.

        Args:
            Image: Sample image to preprocess

        Returns:
            Dict: {
                "ouput": Processed image cropped with mask
                "mask": mask obtained with method 
            }

        '''
        # 94% hsv channel=1
        image_converted = self.color_space(image)
        # Select the channel we are working with from the parameter channel.
        sample_image = image_converted

        if len(image_converted.shape) > 2:
            sample_image = sample_image[:, :, self.channel]

        thr = filters.threshold_otsu(sample_image)
        mask = (sample_image > thr).astype(np.uint8)
        
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 1))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 12))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel_v, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel_h, iterations=1)
        mask_bg = mask.copy()
        cv2.floodFill(mask_bg, None, (0, 0), 1)
        mask = mask - mask_bg
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 1))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 24))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_v, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_h, iterations=1)

        mask = (mask == 0).astype(np.uint8)

        min_area = image.shape[0] * image.shape[1] * 0.05
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        paintings = []

        for contour in contours:
            area = cv2.contourArea(contour)
            box =  cv2.boundingRect(contour)
            y_min, x_min, h, w = box
            x_max, y_max = x_min+w, y_min+h 

            if area < min_area:
                continue
            
            paintings.append((area, (x_min, y_min, x_max, y_max)))

        paintings = sorted(paintings, key=lambda x: x[0], reverse=True)
        paintings = paintings[:2]
        final_mask = np.zeros_like(mask, dtype=np.bool8)

        for area, coords in paintings:
            final_mask[coords[0]:coords[2], coords[1]:coords[3]] = True
        
        return {"result": image.copy(), "mask":  final_mask, "bb": [coords for _, coords in paintings]}


@Registry.register_preprocessing
class PaintThePaintingAdaptativeMaskPreprocessor(Preprocessing):
    name: str = "painter_adaptative_mask_preprocessor"  

    def __init__(self, color_space: str = "hsv", channel: int = 0, metric: str = "std", thr_global: float = 20, fill_holes: bool = True,  **kwargs) -> None:
        self.color_space = TO_COLOR_SPACE[color_space]
        self.channel = channel
        self.metric = METRICS[metric]
        self.thr_global = thr_global
        self.fill_holes = fill_holes

    def run(self,  image, **kwargs) -> Dict[str, np.ndarray]:
        '''
        run(**kwargs) takes an image as an imput and applies morhpology operators to
        create the mask.

        Args:

            Image: Sample image to preprocess

        Returns:
            Dict: {
                "ouput": Processed image cropped with mask
                "mask": mask obtained with method 
            }

        '''
        # 91% F1
        # original_shape = image.shape
        image_converted = self.color_space(image)
        # Select the channel we are working with from the parameter channel.
        sample_image = image_converted

        if len(image_converted.shape) > 2:
            sample_image = sample_image[:, :, self.channel]
        
        mask = cv2.blur(sample_image, (5,5), cv2.BORDER_DEFAULT)
        mask = (cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 11, 2) == 0).astype(np.uint8)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_v, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_h, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel_v, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel_h, iterations=3)
        mask_bg = mask.copy()
        cv2.floodFill(mask_bg, None, (0, 0), 1)
        mask = mask - mask_bg
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 1))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 12))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_v, iterations=12)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_h, iterations=12)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_v, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_h, iterations=2)

        mask = (mask == 0).astype(np.uint8)
        
        min_area = image.shape[0] * image.shape[1] * 0.05
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        paintings = []

        for contour in contours:
            area = cv2.contourArea(contour)
            box =  cv2.boundingRect(contour)
            y_min, x_min, h, w = box
            x_max, y_max = x_min+w, y_min+h 

            if area < min_area:
                continue
            
            paintings.append((area, (x_min, y_min, x_max, y_max)))

        paintings = sorted(paintings, key=lambda x: x[0], reverse=True)
        paintings = paintings[:2]
        final_mask = np.zeros_like(mask, dtype=np.bool8)

        for area, coords in paintings:
            final_mask[coords[0]:coords[2], coords[1]:coords[3]] = True
        
        painting_bbs = [coords for _, coords in paintings]
        painting_bbs = sorted(painting_bbs, key=lambda x: x[1])
        return {"result": image.copy(), "mask":  final_mask, "bb": painting_bbs}


@Registry.register_preprocessing
class LaplacianMaskPreprocessor(Preprocessing):
    name: str = "laplacian_mask_preprocessor"  

    def __init__(self, min_area: float = 0.05,  **kwargs) -> None:
        pass

    def run(self,  image, **kwargs) -> Dict[str, np.ndarray]:
        '''
        run(**kwargs) takes an image as an imput and filters low frequency values
        using the fourier transform.
        Args:
            Image: Sample image to preprocess
        Returns:
            Dict: {
                "ouput": Processed image cropped with mask
                "mask": mask obtained with method 
            }
        '''
        original_shape = image.shape[:2]
        (height, width) = original_shape

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        # diff_image = (rgb*2 - v[:,:,None]).mean(axis=-1)
        diff_image = s

        diff_image = diff_image - diff_image.min()
        diff_image = diff_image / diff_image.max()
        diff_image = (diff_image * 255).astype(np.uint8)

        diff_image = cv2.GaussianBlur(diff_image, (5, 5), 0)

        laplacian_kernel = np.array([[-1,-1,-1],
                                    [-1, 8,-1],
                                    [-1,-1,-1]])
        diff_image = cv2.filter2D(src=diff_image, ddepth=-1, kernel=laplacian_kernel).astype(np.uint8)

        diff_image = (cv2.adaptiveThreshold(diff_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, 11, 2) == 0).astype(np.uint8)

        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        diff_image = cv2.morphologyEx(diff_image, cv2.MORPH_DILATE, kernel_v, iterations=1)
        diff_image = cv2.morphologyEx(diff_image, cv2.MORPH_DILATE, kernel_h, iterations=1)

        min_height = int(original_shape[0] * 0.01)
        min_width = int(original_shape[1] * 0.01)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_height))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (min_width, 1))
        diff_image = cv2.morphologyEx(diff_image, cv2.MORPH_OPEN, kernel_v, iterations=1)
        diff_image = cv2.morphologyEx(diff_image, cv2.MORPH_OPEN, kernel_h, iterations=1)

        contours, hierarchy = cv2.findContours(diff_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        painting_bbs = []
        polygons = []
        filtered_contours = []
        min_area = original_shape[0] * original_shape[1] * 0.05
        min_height = original_shape[0] * 0.1
        min_width = original_shape[1] * 0.1

        for contour in contours:
            y, x, h, w = cv2.boundingRect(contour)
            # contour_area = cv2.contourArea(contour)
            contour_area = w * h

            if contour_area < min_area or h < min_height or w < min_width:
                continue
            
            painting_bbs.append((x, y, x+w, y+h))

            hull = cv2.convexHull(contour)
            polygons.append(hull)
            filtered_contours.append(contour)

        mask = np.zeros_like(diff_image)
        final_polygons = []

        for polygon in polygons:
            # Black image same size as original input:
            hullImg = np.zeros((height, width), dtype=np.uint8)

            # Draw the points:
            cv2.drawContours(hullImg, [polygon], 0, 255, 2)

            hull = [tuple(p[0]) for p in polygon]

            # Find all the corners
            tr = max(hull, key=lambda x: x[0] - x[1])
            tl = min(hull, key=lambda x: x[0] + x[1])
            br = max(hull, key=lambda x: x[0] + x[1])
            bl = min(hull, key=lambda x: x[0] - x[1])

            corner_list = np.array([tl, tr, br, bl])

            final_polygons.append(corner_list)
            mask = cv2.fillConvexPoly(mask, np.array(corner_list, 'int32'), 255)

        return {"result": image.copy(), "mask":  mask > 0, "bb": painting_bbs}


@Registry.register_preprocessing
class HoughMaskPreprocessor(Preprocessing):
    name: str = "hough_mask_preprocessor"  

    def __init__(self, min_area: float = 0.05,  **kwargs) -> None:
        pass

    def painter(self, image):
        image_bg = image.copy()
        last_col = image.shape[1] - 1

        for i in range(0, image.shape[0], image.shape[0] // 8):
            if image[i, 0] == 0:
                cv2.floodFill(image_bg, None, (0, i), 1)
            if image[i, last_col] == 0:
                cv2.floodFill(image_bg, None, (last_col, i), 1)

        new_image = image - image_bg
        new_image = (new_image == 0).astype(np.uint8)
        return new_image

    def run(self,  image, **kwargs) -> Dict[str, np.ndarray]:
        '''
        run(**kwargs) takes an image as an imput and filters low frequency values
        using the fourier transform.
        Args:
            Image: Sample image to preprocess
        Returns:
            Dict: {
                "ouput": Processed image cropped with mask
                "mask": mask obtained with method 
            }
        '''
        original_shape = image.shape[:2]
        (height, width) = original_shape

        image_resized = image_resize(image, 1000, 1000)
        resized_shape = image_resized.shape[:2]

        hsv = cv2.cvtColor(image_resized, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        # diff_image = (rgb*2 - v[:,:,None]).mean(axis=-1)
        diff_image = s

        diff_image = diff_image - diff_image.min()
        diff_image = diff_image / diff_image.max()
        diff_image = (diff_image * 255).astype(np.uint8)

        diff_image = cv2.GaussianBlur(diff_image, (7, 7), 0)

        v = np.median(diff_image)
        # apply automatic Canny edge detection using the computed median
        sigma=0.3
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(diff_image, lower, upper)

        thr = filters.threshold_otsu(edges)
        edges = (edges > thr).astype(np.uint8)

        mask = np.zeros_like(diff_image)
        max_line_gap = int(resized_shape[1] * 0.05)
        linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=20, maxLineGap=max_line_gap)   

        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(mask, (l[0], l[1]), (l[2], l[3]), 255, 3, cv2.LINE_AA)

        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel_v, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel_h, iterations=1)

        mask = self.painter(mask)

        min_x = int(resized_shape[0] * 0.1)
        min_y = int(resized_shape[1] * 0.1)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_x))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (min_y, 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_v, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_h, iterations=1)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        painting_bbs = []
        polygons = []
        min_area = resized_shape[0] * resized_shape[1] * 0.05
        min_height = resized_shape[0] * 0.1
        min_width = resized_shape[1] * 0.1

        for contour in contours:
            x, y, h, w = cv2.boundingRect(contour)
            # contour_area = cv2.contourArea(contour)
            contour_area = w * h

            if contour_area < min_area or h < min_height or w < min_width:
                continue
            
            painting_bbs.append((x, y, x+h, y+w))

            hull = cv2.convexHull(contour)
            polygons.append(hull)

        mask = np.zeros_like(diff_image)
        final_polygons = []

        for polygon in polygons:
            # Black image same size as original input:
            hullImg = np.zeros((height, width), dtype=np.uint8)

            # Draw the points:
            cv2.drawContours(hullImg, [polygon], 0, 255, 2)

            hull = [tuple(p[0]) for p in polygon]

            # Find all the corners
            tr = max(hull, key=lambda x: x[0] - x[1])
            tl = min(hull, key=lambda x: x[0] + x[1])
            br = max(hull, key=lambda x: x[0] + x[1])
            bl = min(hull, key=lambda x: x[0] - x[1])

            corner_list = np.array([tl, tr, br, bl])

            final_polygons.append(corner_list)
            mask = cv2.fillConvexPoly(mask, np.array(corner_list, 'int32'), 255)

        x_scaling = original_shape[0] / resized_shape[0]
        y_scaling = original_shape[1] / resized_shape[1]
        painting_bbs = [(int(x_min * x_scaling), int(y_min * y_scaling), int(x_max * x_scaling),
                     int(y_max * y_scaling)) for x_min, y_min, x_max, y_max in painting_bbs]
        mask = cv2.resize(mask, (original_shape[1], original_shape[0]))

        return {"result": image.copy(), "mask":  mask > 0, "bb": painting_bbs}


@Registry.register_preprocessing
class FourierMaskPreprocessor(Preprocessing):
    name: str = "fourier_mask_preprocessor"  

    def __init__(self, color_space: str = "hsv", channel: int = 1 , **kwargs) -> None:
        self.color_space = TO_COLOR_SPACE[color_space]
        self.channel = channel

    def painter(self, image):
        image_bg = image.copy()
        last_col = image.shape[1] - 1

        for i in range(image.shape[0]):
            if image[i, 0] == 0:
                cv2.floodFill(image_bg, None, (0, i), 1)
                break
            if image[i, last_col] == 0:
                cv2.floodFill(image_bg, None, (last_col, i), 1)
                break

        new_image = image - image_bg
        new_image = (new_image == 0).astype(np.uint8)
        return new_image

    def run(self,  image, **kwargs) -> Dict[str, np.ndarray]:
        '''

        run(**kwargs) takes an image as an imput and filters low frequency values
        using the fourier transform.
        Args:
            Image: Sample image to preprocess

        Returns:
            Dict: {
                "ouput": Processed image cropped with mask
                "mask": mask obtained with method 
            }

        '''
        image_converted = self.color_space(image)
        # Select the channel we are working with from the parameter channel.
        sample_image = image_converted

        if len(sample_image.shape) > 2:
            sample_image = sample_image[:, :, self.channel]

        img = sample_image
        f = np.fft.fft2(img)
        rows, cols = img.shape
        crow,ccol = int(rows/2) , int(cols/2)
        f[crow-1:crow+2, ccol-1:ccol+2] = 0
        img_back = np.fft.ifft2(f)
        img_back = 1 - np.real(img_back)
        img_back = img_back - np.min(img_back)
        img_back = (img_back / np.max(img_back) * 255).astype(np.uint8)
        
        mask = cv2.blur(img_back, (5,5), cv2.BORDER_DEFAULT) 

        # thr = filters.threshold_otsu(mask)
        # mask = (mask > thr).astype(np.uint8)

        mask = (cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 7, 2) == 0).astype(np.uint8)

        # mask = (img_back > 200).astype(np.uint8)

        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_v, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_h, iterations=3)

        mask = self.painter(mask)

        min_area = image.shape[0] * image.shape[1] * 0.05
        min_x = int(image.shape[0] * 0.1)
        min_y = int(image.shape[1] * 0.1)

        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (min_x, 1))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_y))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_v, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_h, iterations=1)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        paintings = []

        for contour in contours:
            area = cv2.contourArea(contour)
            box =  cv2.boundingRect(contour)
            y_min, x_min, h, w = box
            x_max, y_max = x_min+w, y_min+h 

            if area < min_area:
                continue
            
            paintings.append((area, (x_min, y_min, x_max, y_max)))

        paintings = sorted(paintings, key=lambda x: x[0], reverse=True)
        paintings = paintings[:2]
        final_mask = np.zeros_like(mask, dtype=np.bool8)

        for area, coords in paintings:
            final_mask[coords[0]:coords[2], coords[1]:coords[3]] = True
        
        painting_bbs = [coords for _, coords in paintings]
        painting_bbs = sorted(painting_bbs, key=lambda x: x[1])
        return {"result": image.copy(), "mask":  final_mask, "bb": painting_bbs}


@Registry.register_preprocessing
class ContourMaskPreprocessor(Preprocessing):
    name: str = "contour_mask_preprocessor"

    def __init__(self, lower_threshold: int = 150, upper_threshold: int = 100, kernel_size: int = 35, **kwargs) -> None:
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.kernel_size = kernel_size

    def findSignificantContours(self, contours, dim):
        result = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.lower_threshold * self.lower_threshold < area < dim[0]*dim[1] - self.upper_threshold:
                result.append(contour)
        return result

    def run(self,  image, **kwargs) -> Dict[str, np.ndarray]:
        original_shape = image.shape[:2]
        (height, width) = original_shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 10, 100)

        # dilate edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilate = cv2.dilate(edged, kernel, iterations=1)

        # find the significant contours based on minimum & maximum area
        first_contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        significant_contours = self.findSignificantContours(first_contours, image.shape)

        # create mask from significant contours
        temp_mask = np.zeros(image.shape)
        cv2.drawContours(temp_mask, significant_contours, -1, (255, 255, 255), cv2.FILLED)

        # refine mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))
        erosion = cv2.erode(temp_mask, kernel, iterations=2)
        dilation = cv2.dilate(erosion, kernel, iterations=1)

        # search for contours again in order to compose better mask
        temp_mask = cv2.cvtColor(np.uint8(dilation), cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_contours = self.findSignificantContours(contours, image.shape)

        # find coordinates
        painting_bbs = []
        mask = np.zeros_like(temp_mask)
        for contour in significant_contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            rotation = rect[2]

            hullImg = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(hullImg, [box], 0, 255, 1)

            # Find all the corners
            tr = max(box, key=lambda x: x[0] - x[1])
            tl = min(box, key=lambda x: x[0] + x[1])
            br = max(box, key=lambda x: x[0] + x[1])
            bl = min(box, key=lambda x: x[0] - x[1])

            painting_bbs.append((tl[0], tl[1], br[0], br[1]))
            corner_list = np.array([tl, tr, br, bl])
            mask = cv2.fillConvexPoly(mask, np.array(corner_list, 'int32'), 255)

        return {"result": image.copy(), "mask": mask > 0, "bb": painting_bbs, "angle": rotation}
