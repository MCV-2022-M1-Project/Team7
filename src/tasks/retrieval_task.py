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
        final_output_w1=[]
        final_output_w2=[]
        
        logging.info("Carrying out the task...")

        for sample in tqdm(self.query_dataset, total=self.query_dataset.size()):
            image = sample.image
            images_list = []

            for pp in self.preprocessing:
                output = pp.run(image)
                image = output["result"]

                if "mask" in output:
                    image = (image * output["mask"][:,:,None]).astype(np.uint8)

                if "text_mask" in output:
                    image = (image * (1-(output["text_mask"][:,:,None] / 255))).astype(np.uint8)

                if "bb" in output:
                    images_list = []

                    for bb in output["bb"]:
                        images_list.append(image[bb[0]:bb[2], bb[1]:bb[3]])

            if len(images_list) > 0:
                image = images_list

            if type(image) is not list:
                image = [image]

            feats_pred = self.extractor.run(image, tokenizer=self.tokenizer)["result"]
            top_k_pred = neighbors.kneighbors(feats_pred, n_neighbors=self.config.distance.n_neighbors, return_distance=False)
            final_output_w1.append([int(v) for v in top_k_pred[0]])
            final_output_w2.append([[int(v) for v in top_k_pred[i]] for i in range(len(image))])

            if not inference_only:
                for metric in self.metrics:
                    metric.compute(sample.correspondance, top_k_pred)

        if not inference_only:
            logging.info(f"Printing report and saving to disk.")

            for metric in self.metrics:
                logging.info(f"{str(metric.metric)}: {metric.average}")

            write_report(self.report_path, self.config, self.metrics)
        else:
            write_report(self.report_path, self.config)
        
        with open(os.path.join(self.output_dir, "result_w1.pkl"), 'wb') as f:
            pickle.dump(final_output_w1, f)
        
        with open(os.path.join(self.output_dir, "result_w2.pkl"), 'wb') as f:
            pickle.dump(final_output_w2, f)
