import argparse
import os

from src.common.configuration import load_configuration
from src.common.registry import Registry
from src.datasets.dataset import Dataset


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Steganography trainer parser')
    parser.add_argument('--datasets_dir', type=str, default='./datasets',
                        help='location of the dataset')
    parser.add_argument('--config', type=str, default='./config/task1.yaml',
                        help='location of the configuration file')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='training batch size')
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    config = load_configuration(args.config)
    
    for ds in config.datasets:
        dataset = Dataset(os.path.join(args.datasets_dir, ds))
        Registry.register_dataset(ds, dataset)

    Registry.register("selected_metrics", config.metrics)
    Registry.register("task_config", config.task)


if __name__ == "__main__":
    args = __parse_args()
    main(args)
