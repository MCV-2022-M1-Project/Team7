from __future__ import annotations
import os
import numpy as np
import cv2
import pickle
from typing import List, Optional, Tuple
from glob import glob

from dataclasses import dataclass


@dataclass
class Sample:
    id: int
    mask: np.ndarray
    image: np.ndarray
    annotation: Optional[Tuple[str, str]]
    correspondance: Optional[List[int]]
    text_boxes: Optional[List[Tuple[int, int, int, int]]]


class Dataset:
    def __init__(self, path: str, name: str = "default", preload: bool = False) -> None:
        self.name = name
        mask_paths = sorted(glob(os.path.join(path, "*.png")))
        image_paths = sorted(glob(os.path.join(path, "*.jpg")))
        ann_paths = sorted(glob(os.path.join(path, "*.txt")))
        self.__mask_paths = mask_paths
        self.__image_paths = image_paths
        self.size = len(image_paths)

        assert len(image_paths) > 0, f"No images were found on {path}."

        self.__masks: List[Optional[np.ndarray]] = [None for _ in range(self.size)]
        self.__images: List[Optional[np.ndarray]] = [None for _ in range(self.size)]
        self.annotations: List[Tuple[str, str]] = []
        self.correspondances: List[List[int]] = []
        self.text_boxes: List[List[Tuple[int, int, int, int]]] = []

        corresps_path = os.path.join(path, "gt_corresps.pkl")

        if os.path.exists(corresps_path):
            with open(corresps_path, "rb") as f:
                self.correspondances = pickle.load(f)
                assert type(self.correspondances) is list

        text_bb_path = os.path.join(path, "text_boxes.pkl")

        if os.path.exists(text_bb_path):
            with open(text_bb_path, "rb") as f:
                text_boxes_samples: List[List[List[np.ndarray]]] = pickle.load(f)

                for text_box_list in text_boxes_samples:
                    if len(text_box_list[0]) > 2:
                        self.text_boxes.append(text_box_list)
                    else:
                        self.text_boxes.append([(
                            text_box[0][0],
                            text_box[0][1],
                            text_box[2][0],
                            text_box[2][1],
                        ) for text_box in text_box_list])

        if preload:
            self.__load_images()
            self.__load_masks()

        for path in ann_paths:
            with open(path, "r", encoding='latin-1') as f:
                ann = f.read()[2:-2].replace("'", "").split(", ")

            self.annotations.append(tuple(ann))

    def __load_image(self, id: int) -> np.ndarray:
        image = self.__images[id]

        if image is not None:
            return image

        image = cv2.imread(self.__image_paths[id], cv2.IMREAD_COLOR)
        self.__images[id] = image
        return image

    def __load_mask(self, id: int) -> np.ndarray:
        mask = self.__masks[id]

        if mask is not None:
            return mask

        mask = cv2.imread(self.__mask_paths[id], cv2.IMREAD_COLOR)
        self.__masks[id] = mask
        return mask

    def __load_images(self):
        for i, path in enumerate(self.__image_paths):
            if self.__images[i] is None:
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                self.__images[i] = image

    def __load_masks(self):
        for i, path in enumerate(self.__mask_paths):
            if self.__masks[i] is None:
                mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                self.__masks[i] = mask

    @property
    def images(self):
        self.__load_images()
        return self.__images

    @property
    def masks(self):
        self.__load_masks()
        return self.__masks

    def __len__(self) -> int:
        return self.size

    def get_item(self, id: int) -> Sample:
        return self.__getitem__(id)

    def __iter__(self):
        return (self.__getitem__(id) for id in range(len(self)))

    def __getitem__(self, id: int) -> Sample:
        if len(self.__masks) > 0:
            mask = self.__load_mask(id)
        else:
            mask = None

        if len(self.annotations) > 0:
            annotation = self.annotations[id]
        else:
            annotation = None

        if len(self.correspondances) > 0:
            corresp = self.correspondances[id]
        else:
            corresp = None

        if len(self.text_boxes) > 0:
            text_bbs = self.text_boxes[id]
        else:
            text_bbs = None

        return Sample(
            id,
            mask,
            self.__load_image(id),
            annotation,
            corresp,
            text_bbs,
        )
