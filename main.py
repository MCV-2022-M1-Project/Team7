import argparse
import os
import logging
from uuid import uuid4

from src.common.configuration import load_configuration
from src.common.registry import Registry
from src.common.utils import wrap_metric_classes
from src.datasets.dataset import Dataset


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='M1 project parser')
    parser.add_argument('--datasets_dir', type=str, default='./datasets',
                        help='location of the datasets')
    parser.add_argument('--retrieval_ds_dir', type=str, default='./datasets/museum',
                        help='location of the retrieval dataset')
    parser.add_argument('--preload_retrieval_ds', type=bool, default=False,
                        help='whether to preload retrieval images in memory (takes some time)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='location of the output')
    parser.add_argument('--config', type=str, default='./config/retrieval_bad.yaml',
                        help='location of the configuration file')
    parser.add_argument('--inference_only', action="store_true",
                        help='run inference and skip metrics computation')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='number of parallel workers')
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    config = load_configuration(args.config)
    Registry.register("output_dir", args.output_dir)

    logging.info("Loading retrieval dataset...")
    retrieval_dataset = Dataset(args.retrieval_ds_dir, name="retrieval", preload=args.preload_retrieval_ds)
    logging.info("Retrieval dataset loaded.")        

    for task_config in config.tasks:
        id = str(uuid4()).split('-')[0]

        for ds_name in task_config.datasets:
            query_dataset = Dataset(os.path.join(args.datasets_dir, ds_name), name=ds_name)
            # logging.info(f"Registering dataset: '{ds}'.")
            # Registry.register_dataset(ds, query_dataset)

            logging.info(
                f"Running task '{task_config.name}' with id '{id}' on dataset '{ds_name}'.")
            task_class = Registry.get_task_class(task_config.name)

            if "tokenizer" in task_config:
                tokenizer = Registry.get_tokenizer_instance(task_config.tokenizer)
            else:
                tokenizer = None

            if "preprocessing" in task_config:
                preprocessing = Registry.get_preprocessing_instances(
                    task_config.preprocessing)
            else:
                preprocessing = []

            if "features_extractors" in task_config:
                extractors = Registry.get_features_extractor_instances(
                    task_config.features_extractors)
            else:
                extractors = None

            if "metrics" in task_config:
                metrics = Registry.get_metric_instances(task_config.metrics)
                metrics = wrap_metric_classes(metrics)
            else:
                metrics = []

            task = task_class(
                retrieval_dataset,
                query_dataset,
                task_config,
                args.output_dir,
                tokenizer,
                preprocessing,
                extractors,
                metrics,
                id)
            task.run(args.inference_only)
            logging.info("Task completed successfully")


if __name__ == "__main__":
    args = __parse_args()
    main(args)
