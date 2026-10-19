#!/usr/bin/env python3
"""Export a trained checkpoint to ONNX (with a JSON file holding class names and preprocessing).

Example: python export_onnx.py --checkpoint runs/mobilevit_s/best.pt
"""
from plant_cls.config import get_export_parser
from plant_cls.inference import run_export

if __name__ == "__main__":
    run_export(get_export_parser().parse_args())
