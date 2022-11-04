import cv2
import numpy as np
import pytesseract
import itertools
from typing import Dict
import os
import matplotlib.pyplot as plt

from src.common.registry import Registry
from src.common.utils import *
from src.preprocessing.base import Preprocessing
from pytesseract import Output


def visual_templates_loader() -> np.ndarray:
    path = '/'.join(__file__.split('/')[:-1])
    impath = f"{path}/../../tools/abc"
    files = os.listdir(impath)
    for template in filter(lambda x: '.png' in x, files): yield cv2.imread(f"{impath}/{template}", cv2.IMREAD_GRAYSCALE)

@Registry.register_preprocessing
class MorphTextDetector(Preprocessing):
    name: str = "morph_text_detector"

    def run(self,  image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        '''
        Paper: https://www.researchgate.net/publication/50422356_Morphology_Based_Text_Detection_and_Extraction_from_Complex_Video_Scene

        Args:
            Image: Images to detect text on.

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
        opening = cv2.morphologyEx(
            gray, cv2.MORPH_OPEN, kernel_v, iterations=1)
        opening = cv2.morphologyEx(
            opening, cv2.MORPH_OPEN, kernel_h, iterations=1)

        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        closing = cv2.morphologyEx(
            gray, cv2.MORPH_CLOSE, kernel_v, iterations=1)
        closing = cv2.morphologyEx(
            closing, cv2.MORPH_CLOSE, kernel_h, iterations=1)

        diff_img = cv2.absdiff(opening, closing).astype(np.uint8)
        # thr = filters.threshold_otsu(diff_img)
        # binarized = (diff_img > thr).astype(np.uint8)
        binarized = (cv2.adaptiveThreshold(diff_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 13, 6) == 0).astype(np.uint8)

        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        dilated = cv2.morphologyEx(
            binarized, cv2.MORPH_DILATE, kernel_v, iterations=1)
        dilated = cv2.morphologyEx(
            dilated, cv2.MORPH_DILATE, kernel_h, iterations=1)

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

        dilated = dilated[int(dilated.shape[0] * 0.05):int(dilated.shape[0] * 0.95),
                          int(dilated.shape[1] * 0.05):int(dilated.shape[1] * 0.95)]
        contours, hierarchy = cv2.findContours(
            dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
        text_bbs = [(x_min * x_scaling, y_min * y_scaling, x_max * x_scaling,
                     y_max * y_scaling) for x_min, y_min, x_max, y_max in text_blobs]
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
        orig_img = image
        mask = None
        masks = []
        bboxes = []
        # extract each painting's mask
        if mask is None:
            masks = [np.ones(orig_img.shape[:2])]
        else:
            masks = extract_paintings_from_mask(mask)

        # we try to extract one textbox per painting
        for m in masks:
            img = orig_img.copy()
            img[m == 0] = (0, 0, 0)
            sc_br, bbox_br = get_best_textbox_candidate(brightText(img), m)
            sc_dr, bbox_dr = get_best_textbox_candidate(darkText(img), m)
            bbox = bbox_br
            if sc_dr == 0 and sc_br == 0:
                continue
            if sc_dr > sc_br:
                bbox = bbox_dr
            bboxes.append(bbox)

        gen_mask = generate_text_mask(orig_img.shape[:2], bboxes)

        text = []

        for x_min, y_min, x_max, y_max in bboxes:
            cropped_image = image[y_min:y_max, x_min:x_max]
            d = pytesseract.image_to_data(cropped_image, output_type=pytesseract.Output.DICT, lang='cat', nice=5)
            text += d["text"]

        text = " ".join([s for s in text if s != ""])
        return {"result": image.copy(), "text_mask": gen_mask, "text_bb": bboxes, "text": text}


@Registry.register_preprocessing
class TextDetectorWOMask(Preprocessing):
    name: str = "text_detector_wo_mask"

    def __init__(self, blur_size: int = 5, kernel_size=10, kernel_reduction_x=60, kernel_reduction_y=4, *args, **kwargs) -> None:
        self.blur_size = blur_size
        self.kernel_size = kernel_size
        self.kernel_reduction_x = kernel_reduction_x
        self.kernel_reduction_y = kernel_reduction_y

    def run(self,  image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        This function detects the text in the image and returns an array with coordinates of text bbox.

        arg: image in RGB spacecolor.

        return: [x1, y1, x2, y2]

        """
        rgb = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB)  # convert image to RGB color space
        # convert image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # split the channels of the color space in Hue, Saturation and Value
        h, s, v = cv2.split(hsv)
        # TextDetection.find_regions(img)

        # Open morphological transformation using a square kernel with dimensions 10x10
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        s = cv2.GaussianBlur(s, (self.blur_size, self.blur_size), 0)
        # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
        morph_open = cv2.morphologyEx(s, cv2.MORPH_OPEN, kernel)
        # Convert the image to binary
        ret, th1 = cv2.threshold(morph_open, 35, 255, cv2.THRESH_BINARY_INV)

        # Open and close morphological transformation using a rectangle kernel relative to the shape of the image
        shape = image.shape
        kernel = np.ones((shape[0] // self.kernel_reduction_x,
                         shape[1] // self.kernel_reduction_y), np.uint8)
        th2 = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
        #th3 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)

        # Find the contours
        (contours, hierarchy) = cv2.findContours(
            th2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            th3 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)
            #th3 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)
            (contours, hierarchy) = cv2.findContours(
                th3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

         # Find the coordinates of the contours and draw it in the original image
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            c = contours[eval_contours(contours, shape[1])]
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            cv2.drawContours(rgb, [box], 0, (255, 0, 0), 2)
            x = np.array([box[0][0], box[1][0], box[2][0], box[3][0]])
            y = np.array([box[0][1], box[1][1], box[2][1], box[3][1]])
            coordinates = np.array([min(x), min(y), max(x), max(y)])
            mask = np.zeros(th2.shape)
            mask[int(coordinates[1]-5):int(coordinates[3]+5),
                 int(coordinates[0]-5):int(coordinates[2]+5)] = 255
        else:
            coordinates = np.zeros([4])
            mask = (np.zeros(th2.shape)*255).astype(np.uint8)

        # Plot the image
        #titles = ['Original with Bbox']
        #images = [rgb]
        # for i in range(1):
            #plt.subplot(1, 1, i + 1), plt.imshow(images[i], 'gray')
            # plt.title(titles[i])
            #plt.xticks([]), plt.yticks([])
            # plt.show()

        text = ""
        x_min, y_min, x_max, y_max = coordinates
        cropped_image = image[y_min:y_max, x_min:x_max]

        if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
            d = pytesseract.image_to_data(cropped_image, output_type=pytesseract.Output.DICT, lang='cat', nice=5)
            text = " ".join([w for w in d["text"] if len(w) > 1])
            
        final_output = []
        final_output.append([int(v) for v in coordinates])
        return {"result": image.copy(), "text_mask": mask, "text_bb": final_output, "text": text}


@Registry.register_preprocessing
class LaplacianTextDetector(Preprocessing):
    name: str = "laplacian_text_detector"

    def __detect_bb(self, image: np.ndarray, image_gray: np.ndarray, image_hsv: np.ndarray, laplacian_kernel_center_weight: int = 4, morph_open: int = 3, extra_open: int = 0, morh_close: int = 0) -> List[List[int]]:
        pixels_saturated = 1 - (image_hsv > 60)

        laplacian_kernel = np.array([[0,-1,0],
                            [-1,laplacian_kernel_center_weight,-1],
                            [0,-1,0]])
        image_gray = cv2.filter2D(src=image_gray, ddepth=-1, kernel=laplacian_kernel).astype(np.uint8)

        image_gray = cv2.morphologyEx(image_gray, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        image_gray = cv2.morphologyEx(image_gray, cv2.MORPH_OPEN, np.ones((morph_open, morph_open), np.uint8))
        
        if extra_open != 0:
            image_gray = cv2.morphologyEx(image_gray, cv2.MORPH_OPEN, np.ones((extra_open, extra_open), np.uint8))
        
        if morh_close != 0:
            image_gray = cv2.morphologyEx(image_gray, cv2.MORPH_CLOSE, np.ones((morh_close, morh_close), np.uint8))
        
        image_gray = ((image_gray < 210) * 255).astype(np.uint8)
        image_gray = (image_gray * pixels_saturated).astype(np.uint8)

        original_area = np.ceil(image_gray.shape[0] * image_gray.shape[1])
        min_area = int(original_area * 0.01)
        max_area = int(original_area * 0.4)

        contours, hierarchy = cv2.findContours(image_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_blobs = []

        for contour in contours:
            epsilon = 0.04*cv2.arcLength(contour,True)
            approx = cv2.approxPolyDP(contour,epsilon,True)
            contour_area = cv2.contourArea(contour)

            if contour_area < min_area or contour_area > max_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w/h
            rectanglessness = contour_area / (w * h)

            if aspect_ratio < 2.5 or rectanglessness < 0.8:
                continue

            cropped_image = image_gray[y:y+h, x:x+w]

            text_blobs.append((x, y, x+w, y+h))

        # Plot everything
        # plt.subplot(1,2,1)
        # plt.imshow(image)
        # # plt.subplot(1,2,2)
        # # plt.imshow(laplacian_image, cmap='gray')
        # plt.subplot(1,2,2)
        # plt.imshow(image_gray, cmap='gray')
        # plt.show()

        return text_blobs

    def run(self,  image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        '''
        Uses the laplacian to extract the text boxes.
        Uses Tesseract to detect and extract text in a given image.

        Args:
            Image: Image to detect text on.

        Returns:
            Dict: {
                "ouput": Processed image.
                "text_mask": Mask of the text.
                "text_bb": List of bounding box of the text (1 or more per image) [[x1, y1, x2, y2],...]
            }
        '''
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,1]
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # laplacian_kernel_center_weights = [4]
        laplacian_kernel_center_weights = [4, 6, 8, 10, 12, 16]
        # morph_opens = [12]
        morph_opens = [0, 3, 12]
        # extra_opens = [0]
        extra_opens = [0, 3, 12] # 24
        morph_close = [0]
        # morph_close = [0, 3, 12]
        
        mask = np.zeros_like(image_gray)

        for lkcw, mo, eo, mc in itertools.product(*[laplacian_kernel_center_weights, morph_opens, extra_opens, morph_close]):
            # print(lkcw, mo, eo)
            text_blobs = []
            text_blobs += self.__detect_bb(image, image_gray, image_hsv, lkcw, mo, eo, mc)
            text_blobs += self.__detect_bb(image, 255-image_gray, image_hsv, lkcw, mo, eo, mc)

            for x_min, y_min, x_max, y_max in text_blobs:
                mask[y_min:y_max, x_min:x_max] = 1

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        text_blobs = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            text_blobs.append((x, y, x+w, y+h))

        final_bbs = []
        final_mask = np.zeros_like(mask)
        text = ""
        for x_min, y_min, x_max, y_max in text_blobs:
            cropped_image = image[y_min:y_max, x_min:x_max]
            d = pytesseract.image_to_data(cropped_image, output_type=pytesseract.Output.DICT, lang='cat', nice=5)
            new_text = " ".join([w for w in d["text"] if len(w) > 1])

            if len(new_text) > 3 and new_text not in text:
                text += new_text
                final_bbs.append((x_min, y_min, x_max, y_max))
                final_mask[y_min:y_max, x_min:x_max] = 1

        if len(final_bbs) == 0:
            final_bbs = []
        else:
            final_bbs = [final_bbs[0]]

        return {"result": image.copy(), "text_mask": final_mask, "text_bb": final_bbs, "text": text}


@Registry.register_preprocessing
class TheMostStupidTextDetector(Preprocessing):
    name: str = "stupid_text_detector"

    def __init__(self, min_area: float = 0.01, max_area: float = 0.15, *args, **kwargs) -> None:
        self.min_area = min_area
        self.max_area = max_area

    def __find_box(self, image, color, c_range: int = 5):
            selected_color = ((image > color) * (image < color+c_range)).astype(np.uint8)
            selected_color = cv2.morphologyEx(selected_color, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
            selected_color = cv2.morphologyEx(selected_color, cv2.MORPH_OPEN, np.ones((13, 13), np.uint8))

            text_blobs = []
            contours, hierarchy = cv2.findContours(selected_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            original_area = np.ceil(image.shape[0] * image.shape[1])
            min_area = int(original_area * self.min_area)
            max_area = int(original_area * self.max_area)

            for contour in contours:
                contour_area = cv2.contourArea(contour)

                if contour_area < min_area or contour_area > max_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w/h
                rectanglessness = contour_area / (w * h)

                if aspect_ratio < 2.5 or rectanglessness < 0.8:
                    continue

                text_blobs.append((x, y, x+w, y+h))

            return text_blobs

    def run(self,  image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        '''
        Binarizes by many colors to extract de bounding box.
        Uses Tesseract to detect and extract text in a given image.

        Args:
            Image: Image to detect text on.

        Returns:
            Dict: {
                "ouput": Processed image.
                "text_mask": Mask of the text.
                "text_bb": List of bounding box of the text (1 or more per image) [[x1, y1, x2, y2],...]
            }
        '''
        original_shape = image.shape[:2]
        image_resized = image_resize(image, 800, 800)
        # image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,1]
        image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image_gray = (np.array([0.299, 0.587, 0.114]) * image_gray).sum(axis=-1).astype(np.uint8)

        image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)

        mask = np.zeros(original_shape)
        text_blobs = []

        for color in range(0, 255, 2):
            text_blobs += self.__find_box(image_gray, color, 5)
            text_blobs += self.__find_box(image_gray, color, 10)

        x_scaling = original_shape[1] / image_gray.shape[1]
        y_scaling = original_shape[0] / image_gray.shape[0]
        text_blobs = [(
            int(x_min * x_scaling), 
            int(y_min * y_scaling), 
            int(x_max * x_scaling),
            int(y_max * y_scaling)
        ) for x_min, y_min, x_max, y_max in text_blobs]

        text = []
        for x_min, y_min, x_max, y_max in text_blobs:
            # print(x_min, y_min, x_max, y_max)
            mask[y_min:y_max, x_min:x_max] = 1
            cropped_image = image[y_min:y_max, x_min:x_max]
            d = pytesseract.image_to_data(cropped_image, output_type=pytesseract.Output.DICT, lang='cat', nice=5)
            text += d["text"]

        text = " ".join([s for s in text if s != ""])

        return {"result": image.copy(), "text_mask": mask, "text_bb": text_blobs, "text": text}


@Registry.register_preprocessing
class AnywayItsGettingLateTextDetector(Preprocessing):
    name: str = "anyway_text_detector"

    def __init__(self, min_area: float = 0.01, max_area: float = 0.15, *args, **kwargs) -> None:
        self.min_area = min_area
        self.max_area = max_area

    def __find_box(self, image):
        text_blobs = []
        original_area = np.ceil(image.shape[0] * image.shape[1])
        min_area = int(original_area * 0.01)
        max_area = int(original_area * 0.4)
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # contour_area = cv2.contourArea(contour)
            contour_area = w * h

            if contour_area < min_area or contour_area > max_area:
                continue
            
            aspect_ratio = w/h
            rectanglessness = contour_area / (w * h)

            if aspect_ratio < 2.5: # or rectanglessness < 0.6:
                continue

            text_blobs.append((x, y, x+w, y+h))

        return text_blobs

    def run(self,  image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        '''
        Binarizes by many colors to extract de bounding box.
        Uses Tesseract to detect and extract text in a given image.

        Args:
            Image: Image to detect text on.

        Returns:
            Dict: {
                "ouput": Processed image.
                "text_mask": Mask of the text.
                "text_bb": List of bounding box of the text (1 or more per image) [[x1, y1, x2, y2],...]
            }
        '''
        original_shape = image.shape[:2]
        # image_resized = image_resize(image, 800, 800)

        rgb = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB)  # convert image to RGB color space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        diff_image = (rgb * 2 - s[:,:,None] - v[:,:,None] * 2).mean(axis=-1)
        diff_image = cv2.GaussianBlur(diff_image, (5, 5), 0)
        diff_image = cv2.morphologyEx(diff_image, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))    

        diff_image = diff_image - diff_image.min()
        diff_image = diff_image / diff_image.max()
        diff_image = diff_image * 255

        diff_image = cv2.morphologyEx(diff_image, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))    

        text_blobs = []

        for i in range(0, 50, 5):
            text_blobs += self.__find_box((diff_image > (200+i)).astype(np.uint8))
            text_blobs += self.__find_box((diff_image < (55-i)).astype(np.uint8))

        mask = np.zeros_like(diff_image)

        final_bbs = []
        text = ""
        for x_min, y_min, x_max, y_max in text_blobs:
            cropped_image = image[y_min:y_max, x_min:x_max]
            d = pytesseract.image_to_data(cropped_image, output_type=pytesseract.Output.DICT, lang='cat', nice=5)
            new_text = " ".join([w for w in d["text"] if len(w) > 1])

            if len(new_text) > 2 and new_text not in text:
                text += new_text
                final_bbs.append((x_min, y_min, x_max, y_max))
                mask[y_min:y_max, x_min:x_max] = 1
                break

        return {"result": image.copy(), "text_mask": mask, "text_bb": text_blobs, "text": text}

def text_recognition(bbx):
    
    if not bbx.shape[0]*bbx.shape[1]: return ""
    d = pytesseract.image_to_data(bbx, lang = 'cat', nice = 5, output_type=Output.DICT)
    n_boxes = len(d['level'])
    text = []
    
    for i in range(n_boxes):

        detection = d['text'][i]
        if len(detection)!=0: text.append(detection)
    
    return ' '.join(text)



@Registry.register_preprocessing
class HarrisTextDetector(Preprocessing):
    name: str = "harris_text_detector"

    def __init__(self, blur_size = 10, k_size = 5, block_size = 2, ksize_harris = 3, min_area = 3e3, *args, **kwargs) -> None:

        self.blur_size = blur_size
        self.k_size = k_size
        self.block_size = block_size
        self.ksize_harris = ksize_harris
        self.min_area = min_area


    def run(self,  image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:

        image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        res = cv2.cornerHarris(image_gs, self.block_size, self.ksize_harris, 0.0)
        res = cv2.blur(res, [self.blur_size]*2)
        res = 255 * (res - res.min())/(res.max() - res.min())
        thresh = cv2.adaptiveThreshold(res.astype(np.uint8), 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        
        
        kernel = np.ones([self.k_size]*2,np.uint8)
        erosion = cv2.erode(thresh,kernel,iterations = 1)
        thresh = cv2.adaptiveThreshold(erosion.astype(np.uint8), 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        
        black = np.zeros_like(image_gs)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        total_contours = None
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            if self.min_area > h*w or (w<=h): continue 
            if not isinstance(total_contours, np.ndarray): total_contours = c
            else: total_contours = np.vstack([total_contours, c])
            black = cv2.rectangle(black, (x, y), (x + w, y + h), (255, 255, 255), -1)

        x,y,w,h = cv2.boundingRect(total_contours) # This Way Of Doing Boxes Suck So Hard It Should Not Be Needed

        return {
            "result": image.copy(),
            "text_mask": black != 0,
            "text_bb": [(x, y, x+w, y+h)],
            "text": text_recognition(image[y:y+h, x:x+w])
        }
