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
    def __init__(self, path: str, name: str = "default") -> None:
        self.name = name
        mask_paths = sorted(glob(os.path.join(path, "*.png")))
        image_paths = sorted(glob(os.path.join(path, "*.jpg")))
        ann_paths = sorted(glob(os.path.join(path, "*.txt")))

        assert len(image_paths) > 0, f"No images were found on {path}."

        self.masks: List[np.ndarray] = []
        self.images: List[np.ndarray] = []
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
                    self.text_boxes.append([(
                        text_box[0][0],
                        text_box[0][1],
                        text_box[2][0],
                        text_box[2][1],
                    ) for text_box in text_box_list])

        for path in mask_paths:
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.masks.append(mask)

        for path in image_paths:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            self.images.append(image)

        for path in ann_paths:
            with open(path, "r", encoding='latin-1') as f:
                ann = f.read()[2:-2].replace("'", "").split(", ")

            self.annotations.append(tuple(ann))

    def size(self) -> int:
        return len(self.images)

    def get_item(self, id: int) -> Sample:
        return self.__getitem__(id)

    def __iter__(self):
        return (self.__getitem__(id) for id in range(self.size()))

    def __getitem__(self, id: int) -> Sample:
        if len(self.masks) > 0:
            mask = self.masks[id]
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
            self.images[id],
            annotation,
            corresp,
            text_bbs,
        )
