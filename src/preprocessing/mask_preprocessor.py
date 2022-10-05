import cv2
import numpy as np
from typing import Callable, Dict

from src.preprocessing.base import Preprocessing
from src.common.registry import Registry

def sd(sample: np.ndarray)->np.float32: return sample.std()

def fill(img: np.ndarray, kernel: tuple = (10, 10)) -> np.ndarray:

    '''
    Performs filling operation by cv2 openning.
    
    '''

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
    return ~cv2.morphologyEx(~img,cv2.MORPH_OPEN,kernel) 

@Registry.register_preprocessing
class VarianceMaskPreprocessor(Preprocessing):
    name: str = "variance_mask_preprocessor"



    def run(self, image: np.ndarray, channel: int = 0, metric: Callable = sd, thr_global: float = 20, fill_holes: bool = True) -> Dict[str, np.ndarray]:

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

        #TODO: Precondition: Channel first or channel last?
        sample_image = image[:, :, channel] # Select the channel we are working with from the parameter channel. 
        horizontal_mask = np.zeros_like(sample_image) # Create masks for scanning 
        vertical_mask = np.zeros_like(horizontal_mask)
        shape = image.shape 
        
        
        ### Vertical scan ###
        for col in range(shape[0]):

            row_vector = sample_image[col, :] # Extract a particular column
            vertical_mask[col, :] = metric(row_vector) # Set the mask to its metric level
            
            
            
        ### Horizontal scan ###   
        for row in range(shape[1]):
            row_vector = sample_image[:, row]
            horizontal_mask[:, row] = metric(row_vector)
            
        result = (255 * ((vertical_mask > thr_global) * (horizontal_mask > thr_global))).astype(np.uint8) # Perform thresholding to minimum variance required and type to cv2 needs.
        # Perform AND operation in order to calculate intersection of background columns and rows. 
        if fill_holes: result = fill(result) # If fill-holes is set to true, fill the image holes.
        result *= 1

        return {"result": image[result!=0], "mask": result!=0}



@Registry.register_preprocessing
class LocalVarianceMaskPreprocessor(Preprocessing):
    name: str = "local_variance_mask_preprocessor"


    def run(self, image: np.ndarray, channel: int = 0, kernel_size: int = 5, thr_global: float = 5) -> Dict[str, np.ndarray]:

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

        #TODO: Precondition: Channel first or channel last? + Comments
        sample_image = image[:, :, channel]
        shape = sample_image.shape
        mask = np.zeros_like(sample_image)
        
        for step_i in range(0, shape[0], kernel_size):
            for step_j in range(0, shape[1], kernel_size):
                
                patch = sample_image[step_i:step_i+kernel_size, step_j:step_j+kernel_size]
                mask[step_i:step_i+kernel_size, step_j:step_j+kernel_size] = patch.std()

        mask = mask > thr_global


        return {"result": image[mask], "mask": mask}


@Registry.register_preprocessing
class CombinedMaskPreprocessor(Preprocessing):
    name: str = "combined_mask_preprocessor"


    def run(self, image: np.ndarray, channel: int = 0, kernel_size: int = 5, thr_global: float = 20, thr_local: float = 5, fill_holes: bool = True, metric: Callable = sd) -> Dict[str, np.ndarray]:

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

        res_global = VarianceMaskPreprocessor().run(image, channel, metric, thr_global, fill_holes)
        res_local = LocalVarianceMaskPreprocessor().run(image, channel, kernel_size, thr_local)
        mask = (255 * res_global * res_local).astype(np.uint8)
        if fill_holes: fill(mask)
        mask *= 1



        return {"result": image[mask!=0], "mask": mask!=0}

    