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
