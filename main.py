import argparse

from src.datasets.dataset import Dataset


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Steganography trainer parser')
    parser.add_argument('--dataset_dir', type=str, default='./datasets/dev',
                        help='location of the dataset')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='training batch size')
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    dataset = Dataset(args.dataset_dir)


if __name__ == "__main__":
    args = __parse_args()
    main(args)
