#!/usr/bin/env bash
set -euo pipefail

# Baseline YOLO26 training + evaluation on DAWN (no preprocessing filters)
# Usage (repo root):
#   bash scripts/run_baseline.sh
#
# Notes:
# - Works on Linux/Colab. On Windows, run the equivalent python commands in PowerShell.
# - Requires the DAWN dataset to exist at ./dataset/DAWN (default). Override via DAWN_RAW_ROOT.

DAWN_RAW_ROOT=${DAWN_RAW_ROOT:-"dataset/DAWN"}

python src/detection/train_yolo.py \
  --data configs/dawn.yaml \
  --model yolo26n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch auto \
  --device auto \
  --project runs/dawn_baseline \
  --name yolo26_original \
  --seed 42 \
  --raw_root "$DAWN_RAW_ROOT"

python src/detection/evaluate_yolo.py \
  --data configs/dawn.yaml \
  --weights runs/dawn_baseline/yolo26_original/weights/best.pt \
  --imgsz 640 \
  --device auto \
  --out_dir results/baseline \
  --raw_root "$DAWN_RAW_ROOT"
