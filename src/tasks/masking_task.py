import logging
import os
import cv2
from typing import Any
import numpy as np
from tqdm import tqdm


from src.common.registry import Registry
from src.common.utils import wrap_metric_classes, write_report
from src.datasets.dataset import Dataset
from src.tasks.base import BaseTask


@Registry.register_task
class MaskingTask(BaseTask):
    """
    Masking task runner.
    """
    name: str = "masking"

    def run(self, inference_only: bool = False) -> None:
        """

        """
        mask_output_dir = os.path.join(self.output_dir, "masks")
        os.makedirs(mask_output_dir, exist_ok=True)

        for sample in tqdm(self.query_dataset, total=self.query_dataset.size()):
            image = sample.image
            mask_gt = sample.mask
            mask_pred = None

            for pp in self.preprocessing:
                output = pp.run(image)
                image = output["result"]

                if "mask" in output:
                    mask_pred = output["mask"]
                    image = image * np.expand_dims(mask_pred, axis=-1)

            assert mask_pred is not None

            if not inference_only:
                for metric in self.metrics:
                    metric.compute([mask_gt], [mask_pred])

            cv2.imwrite(os.path.join(mask_output_dir,
                        f"{sample.id:05d}.png"), 255*mask_pred)

        if not inference_only:
            logging.info(f"Printing report and saving to disk.")
            for metric in self.metrics:
                logging.info(f"{metric.metric.name}: {metric.average}")

            write_report(self.metrics, self.report_path, self.config)
