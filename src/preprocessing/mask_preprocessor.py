from typing import Dict
from src.preprocessing.base import Preprocessing
from src.common.registry import Registry

import cv2
import numpy as np

def sd(sample: np.ndarray)->np.float32: return sample.std()


@Registry.register_mask_preprocessor
class VarianceMaskPreprocessor(Preprocessing):
    name: str = "variance_mask_preprocessor"

    def fill(self, img: np.ndarray, kernel: tuple = (10, 10)) -> np.ndarray:

        '''
        Performs filling operation by cv2 openning.
        
        '''

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
        return ~cv2.morphologyEx(~img,cv2.MORPH_OPEN,kernel) 

    def run(self, image: np.ndarray, channel: int = 0, metric: function = sd, thr_global: float = 20, fill_holes: bool = True) -> Dict[str, np.ndarray]:

        '''
        
        Image: Sample image to preprocess
        Channel: Channel we are scanning
        Metric: Metric used to calculate the least entropic channel, variance has more predictable behaviour.
        thr_global: Threshold of minimum variance to be considered as possitive sample.
        fill_holes: Boolean. Some cases, when painting is composed by sub-paintings it detects the sub-painting level. 
                    We can solve this in order to adjust to the GT by filling the holes.


        Description:

            run(**kwargs) takes an image as an imput and process standard deviation of each row and column.
            Thresholds and operate boolean and for masking.

            Some asumptions made here: 
                1. Background is on the sides of the image.
                2. Painting is on the center of the image.
                3. Background is the least entropic region (lower variance) of the image. In other words: Walls are more boring than paintings.
                4. Low-entropy in background produces "spike" on histogram, which is characterized by lower variance.
                5. Photo of the painting isn't tilted. Thus, we can scan it iteratively.
        
        
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
        if fill_holes:
            result = self.fill(result) # If fill-holes is set to true, fill the image holes.

        return result



    