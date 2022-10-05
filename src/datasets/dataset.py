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


class Dataset:
    def __init__(self, path: str, name: str = "default") -> None:
        self.name = name
        mask_paths = sorted(glob(os.path.join(path, "*.png")))
        image_paths = sorted(glob(os.path.join(path, "*.jpg")))
        ann_paths = sorted(glob(os.path.join(path, "*.txt")))
        self.masks: List[np.ndarray] = []
        self.images: List[np.ndarray] = []
        self.annotations: List[Tuple[str, str]] = []
        self.correspondances: List[List[int]] = []

        for path in mask_paths:
            mask = cv2.imread(path)
            self.masks.append(mask)

        for path in image_paths:
            image = cv2.imread(path)
            self.images.append(image)

        for path in ann_paths:
            with open(path, "r") as f:
                ann = f.read()[2:-2].replace("'", "").split(", ")

            self.annotations.append(tuple(ann))

        corresps_path = os.path.join(path, "gt_corresps.pkl")

        if os.path.exists(corresps_path):
            with open(corresps_path, "rb") as f:
                corresps = pickle.load(f)
                assert type(corresps) is list

    def size(self) -> int:
        return len(self.images)

    def get_item(self, id: int) -> Sample:
        return self.__getitem__(id)

    def __iter__(self):
        return (self.__getitem__(id) for id in range(self.size()))

    def __getitem__(self, id: int) -> Sample:
        if len(self.annotations) > 0:
            annotation = self.annotations[id]
        else:
            annotation = None

        if len(self.correspondances) > 0:
            corresp = self.correspondances[id]
        else:
            corresp = None

        return Sample(
            id,
            self.masks[id],
            self.images[id],
            annotation,
            corresp,
        )