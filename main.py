import argparse
import os
import logging

from src.common.configuration import load_configuration
from src.common.registry import Registry
from src.datasets.dataset import Dataset


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Steganography trainer parser')
    parser.add_argument('--datasets_dir', type=str, default='./datasets',
                        help='location of the dataset')
    parser.add_argument('--config', type=str, default='./config/masking.yaml',
                        help='location of the configuration file')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='training batch size')
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    config = load_configuration(args.config)
    
    logging.info("Loading datasets.")

    # retrieval_dataset = Dataset(os.path.join(args.datasets_dir, ds), name="retrieval")

    for ds in config.datasets:
        dataset = Dataset(os.path.join(args.datasets_dir, ds), name=ds)
        logging.info(f"Registering dataset: {ds}.")
        Registry.register_dataset(ds, dataset)

    Registry.register("task", config.task)
    task_class = Registry.get_selected_task_class()

    for name, dataset in Registry.get_datasets().items():
        logging.info(f"Running task {config.task.name} in dataset {name}.")
        task = task_class(dataset, config.task)
        task.run()


if __name__ == "__main__":
    args = __parse_args()
    main(args)
