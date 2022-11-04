import logging
import os
import numpy as np
import pickle
import cv2
from typing import Dict
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
                self.tokenizer.fit(
                    [", ".join(l) for ann in self.retrieval_dataset.annotations for l in ann])

        logging.info("Extracting retrieval dataset features...")

        knn_models: Dict[str, NearestNeighbors] = {}

        if self.tokenizer is not None and self.tokenizer.input_type == "str":
            feats_text = self.tokenizer.tokenize([" ".join(l) if type(
                l) is list else l for ann in self.retrieval_dataset.annotations for l in ann])
            distance = Registry.get_distance_instance(
                self.config.tokenizer.distance)
            knn_models[self.tokenizer.name] = NearestNeighbors(
                n_neighbors=self.retrieval_dataset.size, metric=distance.get_reference())
            knn_models[self.tokenizer.name].fit(feats_text)

        if self.extractors is not None:
            for extractor in self.extractors:
                feats = extractor.run(
                    self.retrieval_dataset.images, tokenizer=self.tokenizer)["result"]

                if extractor.name != "sift_features_extractor":
                    distance_config = [
                        e["distance"] for e in self.config.features_extractors if e.name == extractor.name][0]
                    distance = Registry.get_distance_instance(distance_config)
                    knn_models[extractor.name] = NearestNeighbors(
                        n_neighbors=self.retrieval_dataset.size, metric=distance.get_reference())
                    knn_models[extractor.name].fit(feats)
                else:
                    sift_features = feats

        final_output_w1 = []
        final_output_w2 = []

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

                    image = [o["result"] for o in output]
                else:
                    o = pp.run(image)
                    image = o["result"]
                    output = [o]

                if "bb" in output[0]:
                    images_list = []
                    bb_list = output[0]["bb"]

                    for bb in bb_list:
                        images_list.append(image[bb[0]:bb[2], bb[1]:bb[3]])

                    if len(images_list) > 0:
                        image = images_list

                if "text_mask" in output[0]:
                    images_list = []

                    for out in output:
                        img = out["result"]
                        text_mask_pred = out["text_mask"]
                        images_list.append(
                            (img * (1-(text_mask_pred[:, :, None] / 255))).astype(np.uint8))

                    if len(images_list) > 0:
                        image = images_list

                if "text" in output[0]:
                    for out in output:
                        text_transcription.append(out["text"])

                        if self.tokenizer is not None and self.tokenizer.input_type == "str":
                            text_tokens.append(
                                self.tokenizer.tokenize(out["text"])[0])

            if type(image) is not list:
                image = [image]

            rankings = []

            if self.tokenizer is not None and self.tokenizer.input_type == "str":
                assert len(
                    text_transcription) > 0, "If you use a tokenizer you must set a text detection preprocessor!"
                feats = np.array(text_tokens)
                neighs = knn_models[self.tokenizer.name].kneighbors(
                    feats, return_distance=False)
                ranking = np.array([n.argsort() for n in neighs])

                if self.config.tokenizer.feats_w is None:
                    self.config.tokenizer.feats_w = 1.0

                rankings.append(ranking * self.config.tokenizer.feats_w)

            if self.extractors is not None:
                for extractor in self.extractors:
                    feats = extractor.run(
                        image, tokenizer=self.tokenizer)["result"]

                    if extractor.name != "sift_features_extractor":
                        neighs = knn_models[extractor.name].kneighbors(
                            feats, return_distance=False)
                        ranking = np.array([n.argsort() for n in neighs])
                    else:
                        # index_params = dict(algorithm=1, trees=5)
                        # search_params = dict(checks=50)
                        # flann = cv2.FlannBasedMatcher(
                        #     index_params, search_params)
                        ranking = []

                        for query_feat in feats:
                            n_matches_per_sample = []

                            for retr_feat in sift_features:
                                bf = cv2.BFMatcher()
                                matches = bf.knnMatch(query_feat, retr_feat, k=2)

                                if len(matches[0]) == 1:
                                    n_matches_per_sample.append(0)
                                    continue

                                # for m, n in flann.knnMatch(query_feat, retr_feat, k=2):
                                #     if m.distance < 0.7*n.distance:
                                #         matches.append(m)

                                n_matches = len([m for m, n in matches if m.distance < 0.7*n.distance])
                                n_matches_per_sample.append(n_matches)
                            
                            ranking.append(np.array(n_matches_per_sample).argsort()[::-1].argsort())

                        ranking = np.array(ranking)

                        # ranking = np.array([
                        #     np.array([len([m for m, n in flann.knnMatch(query_feat, retr_feat, k=2) if m.distance < 0.7*n.distance])
                        #               for retr_feat in sift_features]).argsort()[::-1].argsort()
                        #     for query_feat in feats
                        # ])

                    w = [e["feats_w"]
                         for e in self.config.features_extractors if e.name == extractor.name][0]

                    if w is None:
                        w = 1.0

                    rankings.append(ranking * w)

            per_image_rankings = []

            for e_ranking in rankings:
                for i, r in enumerate(e_ranking):
                    if len(per_image_rankings) <= i:
                        per_image_rankings.append(r)
                    else:
                        per_image_rankings[i] += r

            top_k_pred = [np.argsort(r) for r in per_image_rankings]

            final_output_w1.append([int(v) for v in top_k_pred[0][:10]])
            final_output_w2.append(
                [[int(v) for v in top_k_pred[i][:10]] for i in range(len(image))])

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
