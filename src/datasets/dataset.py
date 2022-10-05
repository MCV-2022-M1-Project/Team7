import os
import numpy as np
import cv2
from typing import List, Tuple
from glob import glob

from dataclasses import dataclass


@dataclass
class Sample:
    id: int
    mask: np.ndarray
    image: np.ndarray
    annotation: Tuple[str, str]


class Dataset:
    def __init__(self, path: str, name: str = "default") -> None:
        mask_paths = glob(os.path.join(path, "*.png"))
        image_paths = glob(os.path.join(path, "*.jpg"))
        ann_paths = glob(os.path.join(path, "*.txt"))
        self.name = name
        self.masks: List[np.ndarray] = []
        self.images: List[np.ndarray] = []
        self.annotations: List[Tuple[str, str]] = []

        for path in mask_paths:
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.masks.append(mask)

        for path in image_paths:
            image = cv2.imread(path)
            self.images.append(image)

        for path in ann_paths:
            with open(path, "r") as f:
                ann = f.read()[2:-2].replace("'", "").split(", ")

            self.annotations.append(tuple(ann))

    def size(self) -> int:
        return len(self.images)

    def get_item(self, id: int) -> Sample:
        return self.__getitem__(id)

    def __iter__(self):
        return (self.__getitem__(id) for id in range(self.size()))

    def __getitem__(self, id: int) -> Sample:
        return Sample(
            id,
            self.masks[id],
            self.images[id],
            self.annotations[id]
        )