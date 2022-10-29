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
class RetrievalCombTask(BaseTask):
    """
    Base task runner.
    """
    name: str = "retrieval_comb"

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

        features = []
        
        if self.tokenizer is not None and self.tokenizer.input_type == "str":
            feats_text = self.tokenizer.tokenize([" ".join(l) if type(l) is list else l for ann in self.retrieval_dataset.annotations for l in ann])
            features += [np.array(feats_text)]

        if self.extractors is not None:
            features += ([extractor.run(self.retrieval_dataset.images, tokenizer=self.tokenizer)["result"] for extractor in self.extractors])
            features = np.concatenate(features, axis=-1)

        distance = Registry.get_distance_instance(self.config.distance)
        knn_model = NearestNeighbors(n_neighbors=self.config.distance.n_neighbors, metric=distance.get_reference())
        knn_model.fit(features)
        
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

            feats_pred = []

            if self.tokenizer is not None and self.tokenizer.input_type == "str":
                assert len(text_transcription) > 0, "If you use a tokenizer you must set a text detection preprocessor!"
                feats_pred += [np.array(text_tokens)]

            if self.extractors is not None:
                feats_pred += [extractor.run(image, tokenizer=self.tokenizer)["result"] for extractor in self.extractors]

            if len(feats_pred) > 1:
                feats_pred = np.concatenate(feats_pred, axis=-1)
            else:
                feats_pred = feats_pred[0]

            top_k_pred = knn_model.kneighbors(feats_pred, return_distance=False)

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
