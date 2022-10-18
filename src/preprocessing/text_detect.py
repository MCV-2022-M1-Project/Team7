from typing import Dict
import cv2
import numpy as np

from src.common.registry import Registry
from src.common.utils import *
from src.preprocessing.base import Preprocessing


@Registry.register_preprocessing
class MorphTextDetector(Preprocessing):
    name: str = "morph_text_detector"

    def run(self,  image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        '''
        Paper: https://www.researchgate.net/publication/50422356_Morphology_Based_Text_Detection_and_Extraction_from_Complex_Video_Scene

        Args:
            Image: List of images to detect text on.

        Returns:
            Dict: {
                "ouput": Processed image.
                "text_mask": Mask of the text.
                "text_bb": List of bounding box of the text (1 or more per image) [[x1, y1, x2, y2],...]
            }
        '''
        original_shape = image.shape
        image_resized = image_resize(image, 600, 600)
        gray = np.mean(image_resized, axis=-1)
        # gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)[:,:,1].squeeze()

        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_v, iterations=1)
        opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel_h, iterations=1)

        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_v, iterations=1)
        closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel_h, iterations=1)

        diff_img = cv2.absdiff(opening, closing).astype(np.uint8)
        # thr = filters.threshold_otsu(diff_img)
        # binarized = (diff_img > thr).astype(np.uint8)
        binarized = (cv2.adaptiveThreshold(diff_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 13, 6) == 0).astype(np.uint8)

        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        dilated = cv2.morphologyEx(binarized, cv2.MORPH_DILATE, kernel_v, iterations=1)
        dilated = cv2.morphologyEx(dilated, cv2.MORPH_DILATE, kernel_h, iterations=1)

        five = np.ceil(dilated.shape[1] * 0.05)

        for row in range(dilated.shape[0]):
            row_vector = dilated[row, :]
            last_1 = -1
            no_1 = 0
            
            for i, val in enumerate(row_vector):
                if last_1 == -1 and val == 0:
                    continue
                
                if val > 0:
                    if last_1 < five and last_1 > 0:
                        dilated[row, i-last_1:i] = 1

                    last_1 = 0
                    no_1 += 1
                    continue

                last_1 += 1

                if no_1 < five + 4 and no_1 > 0:
                    dilated[row, i-no_1:i] = 0

                no_1 = 0

        dilated = dilated[int(dilated.shape[0] * 0.05):int(dilated.shape[0] * 0.95), int(dilated.shape[1] * 0.05):int(dilated.shape[1] * 0.95)]
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        text_blobs = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < 128*five:
                continue

            x_min = y_min = np.Inf
            x_max = y_max = 0

            for coord in contour:
                x, y = coord[0]

                if x > x_max:
                    x_max = x

                if x < x_min:
                    x_min = x

                if y > y_max:
                    y_max = y

                if y < y_min:
                    y_min = y

            text_blobs.append((x_min, y_min, x_max, y_max))

        mask = np.zeros_like(dilated)

        for x_min, y_min, x_max, y_max in text_blobs:
            mask[y_min:y_max, x_min:x_max] = 1

        x_scaling = original_shape[0] / image_resized.shape[0]
        y_scaling = original_shape[1] / image_resized.shape[1]
        text_bbs = [(x_min * x_scaling, y_min * y_scaling, x_max * x_scaling, y_max * y_scaling) for x_min, y_min, x_max, y_max in text_blobs]
        mask = cv2.resize(mask, (original_shape[1], original_shape[0]))
        return {"result": image.copy(), "text_mask": mask, "text_bb": text_bbs, "text": "not_recognized"}


@Registry.register_preprocessing
class AyanTextDetector(Preprocessing):
    name: str = "ayan_text_detector"

    def run(self,  image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        Given an image (and a mask), extracts the textbox in it, if any found (taking into account a metric).
    
        Parameters
        ----------
        orig_img : image from which extract the textbox
        mask : mask to use. In case of multiple paintings, each connected component will be considered as a different
        painting.
    
        Returns
        -------
        score: bboxes (0 or 1 per connected component found in the mask)
        """
        orig_img=image
        mask=None
        masks = []
        shapes = [] 
        bboxes = []
        # extract each painting's mask
        if mask is None:
            masks = [np.ones(orig_img.shape[:2])]
        else:
            masks = extract_paintings_from_mask(mask)
            
        # we try to extract one textbox per painting
        for m in masks:
            img = orig_img.copy()
            img[m == 0] = (0,0,0)
            sc_br, bbox_br = get_best_textbox_candidate(brightText(img), m)
            sc_dr, bbox_dr = get_best_textbox_candidate(darkText(img), m)
            bbox = bbox_br
            if sc_dr == 0 and sc_br == 0:
                continue
            if sc_dr > sc_br:
                bbox = bbox_dr
            bboxes.append(bbox)
        gen_mask = generate_text_mask(orig_img.shape[:2],bboxes)
        return {"result": image.copy(), "text_mask": gen_mask, "text_bb": bboxes, "text": "not_recognized"}
    
@Registry.register_preprocessing
class TextDetectorWOMask(Preprocessing):
    name: str = "text_detector_wo_mask"
    def __init__(self, *args, **kwargs) -> None:
        return None
    def run(self,  image: np.ndarray, blur_size: int = 5, kernel_size =10, kernel_reduction_x = 60, kernel_reduction_y = 4, **kwargs) -> Dict[str, np.ndarray]:
        """
        This function detects the text in the image and returns an array with coordinates of text bbox.
        
        arg: image in RGB spacecolor.
        
        return: [x1, y1, x2, y2]
        
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert image to RGB color space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # convert image to HSV color space
        h, s, v = cv2.split(hsv)  # split the channels of the color space in Hue, Saturation and Value
        #TextDetection.find_regions(img)
        
        # Open morphological transformation using a square kernel with dimensions 10x10
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        s= cv2.GaussianBlur(s, (blur_size, blur_size), 0)
        # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
        morph_open = cv2.morphologyEx(s, cv2.MORPH_OPEN, kernel)
        # Convert the image to binary
        ret, th1 = cv2.threshold(morph_open, 35, 255, cv2.THRESH_BINARY_INV)
        
        # Open and close morphological transformation using a rectangle kernel relative to the shape of the image
        shape = image.shape
        kernel = np.ones((shape[0] // kernel_reduction_x, shape[1] // kernel_reduction_y), np.uint8)
        th2 = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
        #th3 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)
        
        # Find the contours
        (contours, hierarchy) = cv2.findContours(th2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            th3 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)
            #th3 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)
            (contours, hierarchy) = cv2.findContours(th3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
         # Find the coordinates of the contours and draw it in the original image
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            c = contours[eval_contours(contours, shape[1])]
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            cv2.drawContours(rgb, [box], 0, (255, 0, 0), 2)
            x = np.array([box[0][0],box[1][0],box[2][0],box[3][0]])
            y = np.array([box[0][1],box[1][1],box[2][1],box[3][1]])
            coordinates = np.array([min(x),min(y),max(x),max(y)])
            mask = np.zeros(th2.shape)
            mask[int(coordinates[1]-5):int(coordinates[3]+5), int(coordinates[0]-5):int(coordinates[2]+5)] = 255
        else:
            coordinates = np.zeros([4])
            mask = (np.ones(th2.shape)*255).astype(np.uint8)
       
        #Plot the image
        #titles = ['Original with Bbox']
        #images = [rgb]
        #for i in range(1):
            #plt.subplot(1, 1, i + 1), plt.imshow(images[i], 'gray')
            #plt.title(titles[i])
            #plt.xticks([]), plt.yticks([])
            #plt.show()
        final_output=[]
        final_output.append([int(v) for v in coordinates])
        return {"result": image.copy(), "text_mask": mask, "text_bb": final_output, "text": "not_recognized"}

        
