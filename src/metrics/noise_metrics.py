import numpy as np
from src.common.registry import Registry
from src.metrics.base import Metric
from math import log10, sqrt
from typing import List
from skimage.metrics import structural_similarity as ssim


@Registry.register_metric
class PSNR(Metric):
    name: str = "psnr"

    def compute(self, ground_truth: List[int], predictions: List[int]) -> float:
        mse = np.mean((ground_truth - predictions) ** 2)
        if mse == 0:  # MSE is zero means no noise is present in the signal.
            return 100
        max_pixel = 255.0

        return 20 * log10(max_pixel / sqrt(mse))

    def __str__(self) -> str:
        return f"{self.name}"


@Registry.register_metric
class SSIM(Metric):
    name: str = "ssim"

    def compute(self, ground_truth: List[int], predictions: List[int]) -> float:
        return ssim(im1=ground_truth, im2=predictions, win_size=7, channel_axis=-1)

    def __str__(self) -> str:
        return f"{self.name}"
