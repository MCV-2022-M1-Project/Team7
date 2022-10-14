from typing import Dict, List
import cv2
import os
import numpy as np
import pickle as pkl
import argparse
import time
from tqdm import tqdm

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
        return {"result": image.copy(), "text_mask": mask, "text_bb": text_bbs}


@Registry.register_preprocessing
class AyanTextDetector(Preprocessing):
    name: str = "ayan_text_detector"

    def brightText(self, img):
        """
        Generates the textboxes candidated based on TOPHAT morphological filter.
        Works well with bright text over dark background.

        Parameters
        ----------
        img : ndimage to process

        Returns
        -------
        mask: uint8 mask with regions of interest (possible textbox candidates)
        """
        kernel = np.ones((30, 30), np.uint8)
        img_orig = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

        TH = 150
        img_orig[(img_orig[:, :, 0] < TH) | (img_orig[:, :, 1] < TH)
                | (img_orig[:, :, 2] < TH)] = (0, 0, 0)

        img_orig = closing(img_orig, size=(1, int(img.shape[1] / 8)))

        return (cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY) != 0).astype(np.uint8)


    def darkText(self, img):
        """
        Generates the textboxes candidated based on BLACKHAT morphological filter.
        Works well with dark text over bright background.

        Parameters
        ----------
        img : ndimage to process

        Returns
        -------
        mask: uint8 mask with regions of interest (possible textbox candidates)
        """
        kernel = np.ones((30, 30), np.uint8)
        img_orig = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

        TH = 150
        img_orig[(img_orig[:, :, 0] < TH) | (img_orig[:, :, 1] < TH)
                | (img_orig[:, :, 2] < TH)] = (0, 0, 0)

        img_orig = closing(img_orig, size=(1, int(img.shape[1] / 8)))

        return (cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY) != 0).astype(np.uint8)

    def get_textbox_score(self, m, p_shape):
        """
        Generates a score for how textbox-ish a mask connected component is.

        Parameters
        ----------
        m : mask with the textbox region with 1's
        p_shape : shape of the minimum bounding box enclosing the painting.

        Returns
        -------
        score: score based on size + shape
        """
        m = m.copy()

        # we generate the minimum bounding box for the extracted mask
        x, y, w, h = cv2.boundingRect(m.astype(np.uint8))

        # some upper and lower thresholding depending on its size and the painting size.
        if w < 10 or h < 10 or h > w:
            return 0
        if w >= p_shape[0]*0.8 or h >= p_shape[1]/4:
            return 0

        # we compute the score according to its shape and its size
        sc_shape = np.sum(m[y:y+h, x:x+w]) / (w*h)
        sc_size = (w*h) / (m.shape[0] * m.shape[1])

        final_score = (sc_shape + 50*sc_size) / 2

        return final_score


    def get_best_textbox_candidate(self, mask, original_mask):
        """
        Analyzes all connected components and returns the best one according to the textbox metric.

        Parameters
        ----------
        m : mask with the textboxes regions with 1's
        original_mask : painting mask (size of the whole image)

        Returns
        -------
        score: score based on size + shape
        """
        # we will need it to crop the final textbox region so it does not goes beyond painting limits.
        x, y, w, h = cv2.boundingRect(original_mask.astype(np.uint8))
        p_shape = (w, h)
        p_coords = (x, y)

        # we get the biggest connected component with a score higher than TH as the textbox proposal
        mask_c = mask.copy()
        TH = 0.5
        i = 0
        found = False
        mask = None
        best_sc = 0

        while not found:
            biggest = extract_biggest_connected_component(mask_c).astype(np.uint8)
            
            if np.sum(biggest) == 0:
                return 0, None

            sc = self.get_textbox_score(biggest, p_shape)

            if sc > TH:
                mask = biggest
                best_sc = sc
                found = True
            else:
                mask_c -= biggest

        # we crop it and give it a margin dependant on the painting size.
        x, y, w, h = cv2.boundingRect(mask)
        M_W = 0.05
        M_H = 0.05
        ref = min(p_shape)
        x0, y0, x, y = (x - int(ref*M_W/2), y - int(ref*M_H/2),
                        (x+w) + int(ref*M_W/2), (y+h) + int(ref*M_H/2))
        return best_sc, [max(0, x0), max(0, y0), min(x, p_coords[0] + p_shape[0]), min(y, p_coords[1] + p_shape[1])]

    def run(self,  image: np.ndarray, mask: Optional[np.ndarray] = None, **kwargs) -> Dict[str, np.ndarray]:
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
            img[m == 0] = (0, 0, 0)
            sc_br, bbox_br = self.get_best_textbox_candidate(self.brightText(img), m)
            sc_dr, bbox_dr = self.get_best_textbox_candidate(self.darkText(img), m)
            bbox = bbox_br

            if sc_dr == 0 and sc_br == 0:
                continue

            if sc_dr > sc_br:
                bbox = bbox_dr

            bboxes.append(bbox)

        gen_mask = generate_text_mask(orig_img.shape[:2], bboxes)
        return {"result": image.copy(), "text_mask": gen_mask, "text_bb": bboxes}
