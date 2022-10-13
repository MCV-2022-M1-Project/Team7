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
class TextDetectionTask(BaseTask):
    """
    Text detection task runner.
    """
    name: str = "text_detection"

    def run(self, inference_only: bool = False) -> None:
        """

        """
        mask_output_dir = os.path.join(self.output_dir, "text_masks")
        os.makedirs(mask_output_dir, exist_ok=True)

        for sample in tqdm(self.query_dataset, total=self.query_dataset.size()):
            image = sample.image
            text_bb = sample.text_boxes
            text_boxes_pred = None
            text_mask_pred = None

            for pp in self.preprocessing:
                output = pp.run(image)
                image = output["result"]

                if "text_mask" in output:
                    text_mask_pred = output["text_mask"]

                if "text_bb" in output:
                    text_boxes_pred = output["text_bb"]

            assert text_mask_pred is not None
            assert text_boxes_pred is not None

            if not inference_only:
                for metric in self.metrics:
                    metric.compute([text_bb], [text_boxes_pred])

            cv2.imwrite(os.path.join(mask_output_dir,
                        f"{sample.id:05d}.png"), 255*text_mask_pred)

        if not inference_only:
            logging.info(f"Printing report and saving to disk.")
            for metric in self.metrics:
                logging.info(f"{metric.metric.name}: {metric.average}")

            write_report(self.report_path, self.config, self.metrics)
        else:
            write_report(self.report_path, self.config)
