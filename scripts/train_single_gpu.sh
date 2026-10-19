#!/usr/bin/env bash
# Train on one GPU (or CPU). Usage: bash scripts/train_single_gpu.sh /path/to/dataset [model]
set -euo pipefail

DATA_DIR="${1:?usage: $0 /path/to/dataset [model_name]}"
MODEL="${2:-mobilevit_s}"

python train.py \
  --data_dir "$DATA_DIR" \
  --model "$MODEL" \
  --epochs 30 \
  --batch_size 32 \
  --class_weights
