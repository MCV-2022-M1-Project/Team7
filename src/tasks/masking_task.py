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

    def __init__(self, retrieval_dataset: Dataset, query_dataset: Dataset, config: Any, **kwargs) -> None:
        super().__init__(retrieval_dataset, query_dataset, config, **kwargs)
        self.preprocessing = Registry.get_selected_preprocessing_instances()
        self.metrics = Registry.get_selected_metric_instances()
        self.metrics = wrap_metric_classes(self.metrics)

    def run(self) -> None:
        """

        """
        output_dir = self.config.output_dir
        mask_output_dir = os.path.join(output_dir, "masks")
        report_path = os.path.join(output_dir, f"report_{self.name}_on_{self.query_dataset.name}.txt")
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

            for metric in self.metrics:
                metric.compute([mask_gt], [mask_pred])

            cv2.imwrite(os.path.join(mask_output_dir, f"{sample.id:05d}.png"), 255*mask_pred)

        logging.info(f"Printing report and saving to disk.")
        for metric in self.metrics:
            logging.info(f"{metric.metric.name}: {metric.average}")

        write_report(self.metrics, report_path, self.config)
