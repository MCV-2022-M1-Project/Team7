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

                if not extractor.returns_keypoints:
                    distance_config = [
                        e["distance"] for e in self.config.features_extractors if e.name == extractor.name][0]
                    distance = Registry.get_distance_instance(distance_config)
                    knn_models[extractor.name] = NearestNeighbors(
                        n_neighbors=self.retrieval_dataset.size, metric=distance.get_reference())
                    knn_models[extractor.name].fit(feats)
                else:
                    retr_keypoint_features = feats

        final_output_w1 = []
        final_output_w2 = []
        final_output_frame = []

        logging.info("Carrying out the task...")

        for sample in tqdm(self.query_dataset):
            image = [sample.image]
            images_list = []
            bb_list = []
            text_transcription = []
            text_tokens = []
            frame_output = []

            if self.config.use_gt:
                image = []
                contours, _ = cv2.findContours(self.query_dataset[sample.id].mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    image.append(sample.image[y:y+h, x:x+w])

                if len(image) == 0:
                    image = [sample.image]

            for pp in self.preprocessing:
                output = []

                for img in image:
                    output.append(pp.run(img))

                image = [o["result"] for o in output]

                if "bb" in output[0]:
                    images_list = []
                    bb_list = output[0]["bb"]
                    
                    if "angles" in output[0]:
                        angles = output[0]["angles"]
                        frame_output = []
                        for i, bb in enumerate(bb_list):
                            frame_output.append([angles[i], bb])
                    if "original_angles" in output[0]:
                        for i, bb in enumerate(bb_list):
                            # calculate angle needed for opencv rotation
                            angle = output[0]["original_angles"][i]
                            if angle < -45:
                                angle = -(90 + angle)
                            else:
                                angle = -angle
                            neg_angle = -angle

                            # rotate image only if needed
                            if abs(angle) != 180.0 and abs(angle) != 0.0 and abs(angle) != 90:
                                neg_angle = 90 - neg_angle
                                neg_angle = -neg_angle
                                if neg_angle < -45:
                                    neg_angle = 90 + neg_angle

                                M = cv2.getRotationMatrix2D((image[0].shape[1] // 2, image[0].shape[0] // 2), neg_angle, 1.0)
                                rotated_image = cv2.warpAffine(image[0], M, (image[0].shape[1], image[0].shape[0]),
                                                               flags=cv2.INTER_CUBIC,
                                                               borderMode=cv2.BORDER_REPLICATE)
                                # calculate new corner coordinates
                                bb_points = np.array(bb).reshape((-1, 1, 2))
                                rotated_points = cv2.transform(bb_points, M)

                                tl_new = rotated_points[0][0]
                                bl_new = rotated_points[3][0]
                                tr_new = rotated_points[1][0]

                                images_list.append(rotated_image[tl_new[1]:bl_new[1], tl_new[0]:tr_new[0]])
                            else:
                                images_list.append(image[0][bb[0][1]:bb[3][1], bb[0][0]:bb[1][0]])
                    else:
                        for bb in bb_list:
                            images_list.append(image[0][bb[0]:bb[2], bb[1]:bb[3]])

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

            rankings = []
            in_database = [0 for _ in range(len(image))]

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
                    extractor_config = [
                        e for e in self.config.features_extractors if e.name == extractor.name][0]
                    feats = extractor.run(
                        image, tokenizer=self.tokenizer)["result"]

                    if not extractor.returns_keypoints:
                        dists, neighs = knn_models[extractor.name].kneighbors(
                            feats, return_distance=True)
                        ranking = np.array([n.argsort() for n in neighs])

                        for i, dist in enumerate(dists):
                            # dist = dist / dist.max()
                            # dist = np.tanh(dist)
                            # dist = 1/(1 + np.exp(-dist))

                            if dist[0] <= extractor_config.distance.in_db_thr:
                                in_database[i] += 1

                    else:
                        # index_params = dict(algorithm=0, trees=5)
                        # index_params = dict(algorithm = 6,
                        #                     table_number = 6, # 12
                        #                     key_size = 12,     # 20
                        #                     multi_probe_level = 1) # 20 multi_probe_level = 1)
                        # search_params = dict(checks=50)
                        # matcher = cv2.FlannBasedMatcher(
                        #     index_params, search_params)
                        norm = {
                            "hamming": cv2.NORM_HAMMING,
                            "l2": cv2.NORM_L2
                        }
                        matcher = cv2.BFMatcher(norm[extractor_config.norm])
                        ranking = []

                        for i, query_feat in enumerate(feats):
                            n_matches_per_sample = []

                            for retr_feat in retr_keypoint_features:
                                try:
                                    matches = matcher.knnMatch(query_feat, retr_feat, k=2)
                                except:
                                    matches = [(0, )]

                                if len(matches[0]) == 1:
                                    n_matches_per_sample.append(0)
                                    continue

                                n_matches = len([m[0] for m in matches if len(m) == 2 and m[0].distance < extractor_config.quality_thr*m[1].distance])
                                n_matches_per_sample.append(n_matches / len(matches))

                            ranking.append(np.array(n_matches_per_sample).argsort()[::-1].argsort())

                            # if min(n_matches_per_sample) >= extractor_config.matches_thr:
                            #     in_database[i] += 1
                            if np.sort(n_matches_per_sample)[-1] >= extractor_config.matches_thr:
                                in_database[i] += 1

                        ranking = np.array(ranking)

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

            top_k_pred = []

            for i, r in enumerate(per_image_rankings):
                if in_database[i] < len(self.extractors) / 2:
                    t = [-1]
                else:
                    t = np.argsort(r)

                top_k_pred.append(t)

            final_output_w1.append([int(v) for v in top_k_pred[0][:10]])
            final_output_w2.append(
                [[int(v) for v in top_k_pred[i][:10]] for i in range(len(image))])

            if frame_output:
                final_output_frame.append(frame_output)

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

        with open(os.path.join(self.output_dir, "frames.pkl"), 'wb') as f:
             pickle.dump(final_output_frame, f)
