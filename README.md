# DAWN Baseline (YOLO26)

This repo runs a reproducible baseline object detection experiment on the DAWN (Detection in Adverse Weather Nature) dataset using Ultralytics YOLO.

## Setup (Local)

```bash
python -m pip install -r requirements.txt
```

## Setup (Google Colab Free)

```bash
!pip install -r requirements.txt
```

## Baseline Training (no filters)

```bash
python src/detection/train_yolo.py \
  --data configs/dawn.yaml \
  --model yolo26n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch auto \
  --device auto \
  --project runs/dawn_baseline \
  --name yolo26_original \
  --seed 42
```

## Baseline Evaluation

```bash
python src/detection/evaluate_yolo.py \
  --data configs/dawn.yaml \
  --weights runs/dawn_baseline/yolo26_original/weights/best.pt \
  --imgsz 640 \
  --device auto \
  --out_dir results/baseline
```

## Dataset location

By default, scripts expect the raw DAWN dataset at `dataset/DAWN`. Override with:

- env var: `DAWN_RAW_ROOT=/path/to/DAWN`
- or CLI: `--raw_root /path/to/DAWN`
