#!/usr/bin/env bash
# Train on several GPUs with DistributedDataParallel.
# Usage: NGPUS=4 bash scripts/train_multi_gpu.sh /path/to/dataset [model]
set -euo pipefail

DATA_DIR="${1:?usage: $0 /path/to/dataset [model_name]}"
MODEL="${2:-mobilevit_s}"
NGPUS="${NGPUS:-$(nvidia-smi -L | wc -l)}"

# --batch_size is per GPU; --scale_lr scales the learning rate with the global batch size.
torchrun --nproc_per_node="$NGPUS" train.py \
  --data_dir "$DATA_DIR" \
  --model "$MODEL" \
  --epochs 30 \
  --batch_size 32 \
  --scale_lr \
  --class_weights
