import logging
import os
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


from src.common.registry import Registry
from src.common.utils import write_report
from src.tasks.base import BaseTask


@Registry.register_task
class RetrievalTask(BaseTask):
    """
    Base task runner.
    """
    name: str = "retrieval"

    def run(self, inference_only: bool = False) -> None:
        """

        """
        if self.tokenizer is not None:
            logging.info("Building tokenizer vocabulary...")
            self.tokenizer.fit(self.query_dataset.images)

        logging.info("Extracting retrieval dataset features...")
        feats_retrieval = self.extractor.run(self.retrieval_dataset.images, tokenizer=self.tokenizer)["result"]
        neighbors = NearestNeighbors(n_neighbors=self.config.distance.n_neighbors, metric=self.config.distance.name)
        neighbors.fit(feats_retrieval)
        final_output=[]
        
        logging.info("Carrying out the task...")

        for sample in tqdm(self.query_dataset, total=self.query_dataset.size()):
            image = sample.image
            mask_pred = None

            for pp in self.preprocessing:
                output = pp.run(image)
                image = output["result"]

                if "mask" in output:
                    mask_pred = output["mask"]
                    image = (image * np.expand_dims(mask_pred, axis=-1)).astype(np.uint8)

            feats_pred = self.extractor.run([image], tokenizer=self.tokenizer)["result"]
            top_k_pred = neighbors.kneighbors(feats_pred, n_neighbors=self.config.distance.top_k, return_distance=False)[0]
            final_output.append([int(v) for v in top_k_pred])

            if not inference_only:
                for metric in self.metrics:
                    metric.compute([sample.correspondance], [top_k_pred])

        if not inference_only:
            logging.info(f"Printing report and saving to disk.")

            for metric in self.metrics:
                logging.info(f"{metric.metric.name}: {metric.average}")

            write_report(self.metrics, self.report_path, self.config)
        
        with open(os.path.join(self.output_dir, "result.pkl"), 'wb') as f:
            pickle.dump(final_output, f)
