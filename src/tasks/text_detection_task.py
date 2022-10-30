import logging
import os
import cv2
import pickle
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
            image = sample.image
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
                if type(image) is list:
                    output = []

                    for img in image:
                        output.append(pp.run(img))
                else:
                    output = [pp.run(image)]

                if "bb" in output[0]: # W! Output of painting masks are inverted: (y, x, y2, x2)
                    images_list = []
                    bb_list = output[0]["bb"]

                    for bb in bb_list:
                        images_list.append(image[bb[0]:bb[2], bb[1]:bb[3]])

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
                    
                    final_output.append(text_boxes_pred)

                if "text" in output[0]:
                    for out in output:
                        text_transcription.append(out["text"])

                        if self.tokenizer is not None:
                            text_tokens.append(self.tokenizer.tokenize(out["text"])[0])

            if len(bb_list) > 0:
                trans_corrected_bbs = [(
                        image_bb[1] + text_bb[0],
                        image_bb[0] + text_bb[1],
                        image_bb[1] + text_bb[2],
                        image_bb[0] + text_bb[3],
                    ) for image_bb, text_bb in zip(bb_list, text_boxes_pred)]
                text_boxes_pred = trans_corrected_bbs

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
