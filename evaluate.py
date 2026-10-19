#!/usr/bin/env python3
"""Evaluate a trained checkpoint on an image-folder dataset (<class>/images).

Example: python evaluate.py --checkpoint runs/mobilevit_s/best.pt --data_dir /path/to/test
"""
from plant_cls.config import get_eval_parser
from plant_cls.inference import run_evaluate

if __name__ == "__main__":
    run_evaluate(get_eval_parser().parse_args())
