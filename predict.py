#!/usr/bin/env python3
"""Predict the disease class of one image or a folder of images.

Example: python predict.py --checkpoint runs/mobilevit_s/best.pt --input leaf.jpg --topk 3
"""
from plant_cls.config import get_predict_parser
from plant_cls.inference import run_predict

if __name__ == "__main__":
    run_predict(get_predict_parser().parse_args())
