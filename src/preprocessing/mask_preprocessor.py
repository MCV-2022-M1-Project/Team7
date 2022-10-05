import cv2
import numpy as np
from typing import Callable, Dict

from src.preprocessing.base import Preprocessing
from src.common.registry import Registry

def sd(sample: np.ndarray)->np.float32: return sample.std()


@Registry.register_preprocessing
class VarianceMaskPreprocessor(Preprocessing):
    name: str = "variance_mask_preprocessor"

    #TODO: Typing, if you are reading this, I have to type and comment, but I'm in a meeting.
    def fill(self, img, kernel = (10, 10)):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
        return ~cv2.morphologyEx(~img,cv2.MORPH_OPEN,kernel)

    def run(self, image: np.ndarray, channel: int = 0, metric: Callable = sd, thr_global: float = 20, fill_holes: bool = True) -> Dict[str, np.ndarray]:

        '''
        
        Image:
        Channel:
        Metric:
        thr_global:
        fill_holes:


        Description:

            run(**kwargs) takes an image as an imput and process standard deviation of each row and column.
            Thresholds and operate boolean and for masking.

            Some asumptions made here: Background is on the sides of the image, and it's not much detailed in terms of different colors (histogram is a delta pulse).

            Painting is on the center.

            Photo of the painting isn't tilted.
        
        
        '''

        sample_image = image[:, :, channel]
        horizontal_mask = np.zeros_like(sample_image)
        vertical_mask = np.zeros_like(horizontal_mask)
        shape = image.shape
        
        
        ### Vertical scan ###
        for col in range(shape[0]):

            row_vector = sample_image[col, :]
            vertical_mask[col, :] = metric(row_vector)
            
            
            
        ### Horizontal scan ###   
        for row in range(shape[1]):
            row_vector = sample_image[:, row]
            horizontal_mask[:, row] = metric(row_vector)
            
        result = (255 * ((vertical_mask > thr_global) * (horizontal_mask > thr_global))).astype(np.uint8)
        if fill_holes:
            result = self.fill(result)

        return {"result": result}



    