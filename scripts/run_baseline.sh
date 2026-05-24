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

# Evaluate baseline model
python src/detection/evaluate_yolo.py \
  --data configs/dawn.yaml \
  --model yolo26n.pt \
  --imgsz 640 \
  --device auto \
  --out_dir results/baseline \
  --raw_root "$DAWN_RAW_ROOT"
