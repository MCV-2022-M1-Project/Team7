import logging

from tqdm import tqdm
import os
import cv2

from src.common.registry import Registry
from src.common.utils import write_report, estimate_noise
from src.tasks.base import BaseTask


@Registry.register_task
class DenoisingTask(BaseTask):
    """
    Base task runner.
    """
    name: str = "denoising"

    def run(self, inference_only: bool = False) -> None:
        """

        """
        output_dir = os.path.join(self.output_dir, "denoised")
        os.makedirs(output_dir, exist_ok=True)

        for sample in tqdm(self.query_dataset):
            image = sample.image

            for pp in self.preprocessing:
                if pp.name == "denoise_preprocessor":
                    # only run denoising based on image quality
                    if estimate_noise(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) > 10:
                        logging.info(f"apply denoise on {sample.id}")
                        output = pp.run(image)
                        image = output["result"]

                else:
                    output = pp.run(image)
                    image = output["result"]

            cv2.imwrite(os.path.join(output_dir,
                                     f"{sample.id:05d}.jpg"), image)

            assert image is not None

            if not inference_only:
                for metric in self.metrics:
                    metric.compute(sample.denoised_image, image)

        if not inference_only:
            logging.info(f"Printing report and saving to disk.")

            for metric in self.metrics:
                logging.info(f"{str(metric.metric)}: {metric.average}")

            write_report(self.report_path, self.config, self.metrics)
        else:
            write_report(self.report_path, self.config)
