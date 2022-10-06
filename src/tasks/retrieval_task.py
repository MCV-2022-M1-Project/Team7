import logging
import os
import numpy as np
from typing import Any
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


from src.common.registry import Registry
from src.common.utils import wrap_metric_classes, write_report
from src.datasets.dataset import Dataset
from src.tasks.base import BaseTask


@Registry.register_task
class RetrievalTask(BaseTask):
    """
    Base task runner.
    """
    name: str = "retrieval"

    def __init__(self, retrieval_dataset: Dataset, query_dataset: Dataset, config: Any, **kwargs) -> None:
        super().__init__(retrieval_dataset, query_dataset, config, **kwargs)
        self.preprocessing = Registry.get_selected_preprocessing_instances()
        self.extractor = Registry.get_selected_features_extractor_instance()
        self.metrics = Registry.get_selected_metric_instances()
        self.metrics = wrap_metric_classes(self.metrics)

    def run(self) -> None:
        """

        """
        output_dir = self.config.output_dir
        mask_output_dir = os.path.join(output_dir, "masks")
        report_path = os.path.join(output_dir, f"report_{self.name}_on_{self.query_dataset.name}.txt")
        os.makedirs(mask_output_dir, exist_ok=True)

        feats_retrieval = self.extractor.run(self.retrieval_dataset.images)["result"]
        neighbors = NearestNeighbors(n_neighbors=self.config.features_extractor.n_neighbors)
        neighbors.fit(feats_retrieval)

        for sample in tqdm(self.query_dataset, total=self.query_dataset.size()):
            image = sample.image
            mask_gt = sample.mask
            mask_pred = None

            for pp in self.preprocessing:
                output = pp.run(image)
                image = output["result"]

                if "mask" in output:
                    mask_pred = output["mask"]
                    # TODO: Apply mask to image

            feats_pred = self.extractor.run(np.array([image]))["result"]
            top_k_pred = neighbors.kneighbors(feats_pred, n_neighbors=self.config.features_extractor.top_k, return_distance=False)[0]

            for metric in self.metrics:
                metric.compute([sample.correspondance], [top_k_pred])

        logging.info(f"Printing report and saving to disk.")

        for metric in self.metrics:
            logging.info(f"{metric.metric.name}: {metric.average}")

        write_report(self.metrics, report_path, self.config)
