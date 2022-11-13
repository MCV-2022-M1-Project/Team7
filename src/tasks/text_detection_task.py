import logging
import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm


from src.common.registry import Registry
from src.common.utils import write_report
from src.tasks.base import BaseTask


@Registry.register_task
class TextDetectionTask(BaseTask):
    """
    Text detection task runner.
    """
    name: str = "text_detection"

    def run(self, inference_only: bool = False) -> None:
        if self.tokenizer is not None:
            logging.info("Building tokenizer vocabulary...")
            self.tokenizer.fit([" ".join(l) for ann in self.retrieval_dataset.annotations for l in ann])

        mask_output_dir = os.path.join(self.output_dir, "text_masks")
        text_transcriptions_output_dir = os.path.join(self.output_dir, "text_transcriptions")
        os.makedirs(mask_output_dir, exist_ok=True)
        os.makedirs(text_transcriptions_output_dir, exist_ok=True)
        final_output = []

        for sample in tqdm(self.query_dataset):
            image = [sample.image]
            annotation = sample.annotation

            if self.tokenizer is not None and annotation is not None:
                annotation_tokenized = [self.tokenizer.tokenize(ann) for ann in annotation]
            
            bb_list = []
            text_bb = sample.text_boxes
            text_boxes_pred = []
            text_mask_pred = None
            text_transcription = []
            text_tokens = []

            for pp in self.preprocessing:
                output = []

                for img in image:
                    output.append(pp.run(img))

                image = [o["result"] for o in output]

                if "bb" in output[0]: # W! Output of painting masks are inverted: (y, x, y2, x2)
                    images_list = []
                    bb_list = output[0]["bb"]
                    new_bb_list = []
                    
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
                                new_bb_list.append([tl_new[0], tl_new[1], bl_new[0], bl_new[1]])
                            else:
                                images_list.append(image[0][bb[0][1]:bb[3][1], bb[0][0]:bb[1][0]])
                                new_bb_list.append([bb[0][0], bb[0][1], bb[1][0], bb[3][1]])

                        bb_list = new_bb_list
                    else:
                        for bb in bb_list:
                            images_list.append(image[0][bb[0]:bb[2], bb[1]:bb[3]])

                    if len(images_list) > 0:
                        image = images_list

                if "text_mask" in output[0]:
                    for i, out in enumerate(output):
                        text_mask_pred = out["text_mask"]
                        cv2.imwrite(os.path.join(mask_output_dir,
                            f"{sample.id:05d}_{i}.png"), 255*text_mask_pred)

                if "text_bb" in output[0]:                     
                    for out in output:
                        if len(out["text_bb"]) > 0:
                            text_boxes_pred.append(out["text_bb"][0])

                if "text" in output[0]:
                    for out in output:
                        text_transcription.append(out["text"])

                        if self.tokenizer is not None:
                            text_tokens.append(self.tokenizer.tokenize(out["text"])[0])

            if len(bb_list) > 0:
                trans_corrected_bbs = [[
                        image_bb[0] + text_bb[0],
                        image_bb[1] + text_bb[1],
                        image_bb[0] + text_bb[2],
                        image_bb[1] + text_bb[3],
                    ] for image_bb, text_bb in zip(bb_list, text_boxes_pred)]
                text_boxes_pred = trans_corrected_bbs

            final_output.append(text_boxes_pred)

            if not inference_only:
                for metric in self.metrics:
                    if metric.metric.input_type == "str":
                        metric.compute(annotation, text_transcription) 
                    elif metric.metric.input_type == "token" and self.tokenizer is not None:
                        metric.compute(annotation_tokenized, text_tokens) 
                    elif metric.metric.input_type == "bb":
                        metric.compute([text_bb], [text_boxes_pred])

            if len(text_boxes_pred) == 0:
                text_boxes_pred.append([0,0,0,0])    

            with open(os.path.join(text_transcriptions_output_dir, f"{sample.id:05d}.txt"), 'w', encoding="utf-8") as f:
                f.write("\n".join(text_transcription))     

        if not inference_only:
            logging.info(f"Printing report and saving to disk.")

            for metric in self.metrics:
                logging.info(f"{metric.metric.name}: {metric.average}")

            write_report(self.report_path, self.config, self.metrics)
        else:
            write_report(self.report_path, self.config)

        with open(os.path.join(self.output_dir, "text_boxes.pkl"), 'wb') as f:
            pickle.dump(final_output, f)
