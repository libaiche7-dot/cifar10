"""Entry point: python train.py --config configs/base.yaml"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError("wire up src/ modules here")


if __name__ == "__main__":
    main()
