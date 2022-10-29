import logging
import os
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import cv2 as cv

from src.common.registry import Registry
from src.common.utils import write_report, estimate_noise
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
            if self.tokenizer.input_type == "image":
                self.tokenizer.fit(self.query_dataset.images)
            elif self.tokenizer.input_type == "str":
                self.tokenizer.fit([", ".join(l) for ann in self.retrieval_dataset.annotations for l in ann])

        logging.info("Extracting retrieval dataset features...")

        if self.tokenizer is not None and self.tokenizer.input_type == "str":
            assert self.config.text_distance is not None
            feats_retrieval = self.tokenizer.tokenize([" ".join(l) if type(l) is list else l for ann in self.retrieval_dataset.annotations for l in ann])
            distance = Registry.get_distance_instance(self.config.text_distance)
            text_neighbors = NearestNeighbors(n_neighbors=self.config.text_distance.n_neighbors, metric=distance.get_reference())
            text_neighbors.fit(feats_retrieval)

        if self.extractor is not None:
            distance = Registry.get_distance_instance(self.config.distance)
            feats_retrieval = self.extractor.run(self.retrieval_dataset.images, tokenizer=self.tokenizer)["result"]
            image_neighbors = NearestNeighbors(n_neighbors=self.config.distance.n_neighbors, metric=distance.get_reference())
            image_neighbors.fit(feats_retrieval)
        
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
                        # only run denoising based on image quality
                        if pp.name == "denoise_preprocessor":
                            if estimate_noise(cv.cvtColor(image, cv.COLOR_BGR2GRAY)) > 10:
                                output.append(pp.run(img))
                        else:
                            output.append(pp.run(img))
                else:
                    # only run denoising based on image quality
                    if pp.name == "denoise_preprocessor":
                        if estimate_noise(cv.cvtColor(image, cv.COLOR_BGR2GRAY)) > 10:
                            output = [pp.run(image)]
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

            top_k_pred = {}

            if self.extractor is not None:
                feats_pred = self.extractor.run(image, tokenizer=self.tokenizer)["result"]
                top_k_pred["extractor"] = image_neighbors.kneighbors(feats_pred, return_distance=False)

            if self.tokenizer is not None and self.tokenizer.input_type == "str":
                assert len(text_transcription) > 0, "If you use a tokenizer you must set a text detection preprocessor!"
                top_k_pred["text"] = text_neighbors.kneighbors(text_tokens, return_distance=False)

            if self.extractor is not None:
                if self.tokenizer is not None and self.tokenizer.input_type == "str":
                    top_k_pred = [[int(v) for v in top_k_pred["extractor"][i] if v in top_k_pred["text"][i]] for i in range(len(image))]
                else:
                    top_k_pred = top_k_pred["extractor"]
            elif self.tokenizer is not None and self.tokenizer.input_type == "str":
                top_k_pred = top_k_pred["text"]
            else:
                raise Exception("You are not extracting any features.")

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
