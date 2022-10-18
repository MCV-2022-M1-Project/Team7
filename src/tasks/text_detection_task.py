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
        """

        """
        mask_output_dir = os.path.join(self.output_dir, "text_masks")
        text_transcriptions_output_dir = os.path.join(self.output_dir, "text_transcriptions")
        os.makedirs(mask_output_dir, exist_ok=True)
        os.makedirs(text_transcriptions_output_dir, exist_ok=True)
        final_output = []

        for sample in tqdm(self.query_dataset):
            image = sample.image
            text_bb = sample.text_boxes
            text_boxes_pred = []
            text_mask_pred = None
            text_transcription = []

            for pp in self.preprocessing:
                if type(image) is list:
                    output = []

                    for i, img in enumerate(image):
                        output.append(pp.run(img))
                else:
                    output = [pp.run(image)]

                if "bb" in output[0]:
                    images_list = []

                    for bb in output[0]["bb"]:
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
                        text_boxes_pred.append(out["text_bb"][0])
                    
                    final_output.append(text_boxes_pred)

                if "text" in output[0]:
                    for out in output:
                        text_transcription.append(out["text"])

            if not inference_only:
                for metric in self.metrics:
                    metric.compute([text_bb], [text_boxes_pred])

            if len(text_boxes_pred) == 0:
                text_boxes_pred.append([0,0,0,0])    

            with open(os.path.join(text_transcriptions_output_dir, f"{sample.id:05d}.txt"), 'w') as f:
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
