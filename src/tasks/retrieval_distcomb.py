import logging
import os
from typing import Dict
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from src.common.registry import Registry
from src.common.utils import write_report
from src.tasks.base import BaseTask


@Registry.register_task
class RetrievalDistCombTask(BaseTask):
    """
    Base task runner.
    """
    name: str = "retrieval_distcomb"

    def run(self, inference_only: bool = False) -> None:
        """

        """
        if self.tokenizer is not None:
            logging.info("Building tokenizer vocabulary...")
            if self.tokenizer.input_type == "image":
                self.tokenizer.fit(self.query_dataset.images)
            elif self.tokenizer.input_type == "str":
                self.tokenizer.fit([", ".join(l) for ann in self.retrieval_dataset.annotations for l in ann])

        logging.info("Extracting retrieval dataset features...")

        knn_models: Dict[str, NearestNeighbors] = {}
        
        if self.tokenizer is not None and self.tokenizer.input_type == "str":
            feats_text = self.tokenizer.tokenize([" ".join(l) if type(l) is list else l for ann in self.retrieval_dataset.annotations for l in ann])
            distance = Registry.get_distance_instance(self.config.tokenizer.distance)
            knn_models[self.tokenizer.name] = NearestNeighbors(n_neighbors=self.retrieval_dataset.size, metric=distance.get_reference())
            knn_models[self.tokenizer.name].fit(feats_text)

        if self.extractors is not None:
            for extractor in self.extractors:
                feats = extractor.run(self.retrieval_dataset.images, tokenizer=self.tokenizer)["result"]
                distance_config = [e["distance"] for e in self.config.features_extractors if e.name == extractor.name][0]
                distance = Registry.get_distance_instance(distance_config)
                knn_models[extractor.name] = NearestNeighbors(n_neighbors=self.retrieval_dataset.size, metric=distance.get_reference())
                knn_models[extractor.name].fit(feats)
        
        final_output_w1=[]
        final_output_w2=[]
        
        logging.info("Carrying out the task...")

        for sample in tqdm(self.query_dataset):
            image = sample.image
            images_list = []
            bb_list = []
            text_transcription = []
            text_tokens = []

            for pp in self.preprocessing:
                if type(image) is list:
                    output = []

                    for img in image:
                        output.append(pp.run(img))
                else:
                    output = [pp.run(image)]

                if "bb" in output[0]:
                    images_list = []
                    bb_list = output[0]["bb"]

                    for bb in bb_list:
                        images_list.append(image[bb[0]:bb[2], bb[1]:bb[3]])

                    if len(images_list) > 0:
                        image = images_list

                if "text_mask" in output[0]:
                    images = []

                    for out in output:
                        image = out["result"]
                        text_mask_pred = out["text_mask"]
                        images.append((image * (1-(text_mask_pred[:,:,None] / 255))).astype(np.uint8))

                if "text" in output[0]:
                    for out in output:
                        text_transcription.append(out["text"])

                        if self.tokenizer is not None and self.tokenizer.input_type == "str":
                            text_tokens.append(self.tokenizer.tokenize(out["text"])[0])

            if len(images_list) > 0:
                image = images_list

            if type(image) is not list:
                image = [image]

            distances = []

            if self.tokenizer is not None and self.tokenizer.input_type == "str":
                assert len(text_transcription) > 0, "If you use a tokenizer you must set a text detection preprocessor!"
                feats = np.array(text_tokens)
                dist, neighs = knn_models[self.tokenizer.name].kneighbors(feats, return_distance=True)
                dist = np.array([d[n.argsort()] for d, n in zip(dist, neighs)])
                distances.append(dist * self.config.tokenizer.feats_w)

            if self.extractors is not None:
                for extractor in self.extractors:
                    feats = extractor.run(image, tokenizer=self.tokenizer)["result"]
                    dist, neighs = knn_models[extractor.name].kneighbors(feats, return_distance=True)
                    w = [e["feats_w"] for e in self.config.features_extractors if e.name == extractor.name][0]
                    dist = np.array([d[n.argsort()] for d, n in zip(dist, neighs)])
                    distances.append(dist * w)

            per_image_dists = []

            for e_dists in distances:
                for i, dists in enumerate(e_dists):
                    if len(per_image_dists) <= i:
                        per_image_dists.append(dists)
                    else:
                        per_image_dists[i] += dists

            top_k_pred = [np.argsort(d) for d in per_image_dists]

            final_output_w1.append([v for v in top_k_pred[0][:10]])
            final_output_w2.append([[v for v in top_k_pred[i][:10]] for i in range(len(image))])

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
