#!/usr/bin/env python3
"""Train a MobileViT plant-disease classifier. See README.md for details.

Single GPU / CPU:  python train.py --data_dir /path/to/data
Multi-GPU (DDP):   torchrun --nproc_per_node=4 train.py --data_dir /path/to/data
"""
from plant_cls.config import get_train_parser
from plant_cls.engine import run_training

if __name__ == "__main__":
    run_training(get_train_parser().parse_args())
