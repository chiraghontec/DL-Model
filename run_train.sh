#!/usr/bin/env bash
# Example run script for train.py

DATA_ROOT="/Users/vinayakprasad/Documents/Major Project/Dataset for Crop Pest and Disease Detection"
CHECKPOINT_DIR="/Users/vinayakprasad/Documents/Major Project/ml model 3/checkpoints"

python3 "$(dirname "$0")/train.py" \
  --data_root "$DATA_ROOT" \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-3 \
  --pretrained \
  --save_dir "$CHECKPOINT_DIR" \
  --amp
