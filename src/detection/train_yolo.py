"""Train YOLO26 (Ultralytics) baseline on DAWN (no preprocessing).

Required CLI (example):
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

Behavior:
- Ensures the DAWN dataset is prepared as Ultralytics-YOLO format under data/processed/dawn_yolo.
- Tries to load YOLO26 weights/model via Ultralytics.
- If YOLO26 is unavailable, prints ultralytics version and guidance.
- Fallback model is only used when --allow_fallback is enabled.

Notes:
- Default raw dataset root: ./dataset/DAWN (override via --raw_root or env DAWN_RAW_ROOT).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script: `python src/detection/train_yolo.py ...`
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dawn_prepare import BASELINE_CLASSES, prepare_dawn_yolo
from src.utils.paths import dawn_processed_root, dawn_raw_root
from src.utils.seed import set_seed
from src.utils.ultralytics_data import load_data_yaml
from src.utils.yolo26 import try_load_ultralytics_model


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--data", required=True, help="Ultralytics data.yaml (e.g., configs/dawn.yaml)")
    p.add_argument("--model", required=True, help="Model spec or weights path (e.g., yolo26n.pt)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", default="auto")
    p.add_argument("--device", default="auto")
    p.add_argument("--project", default="runs/dawn_baseline")
    p.add_argument("--name", default="yolo26_original")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--raw_root",
        default=None,
        help="Raw DAWN dataset root (default: env DAWN_RAW_ROOT or ./dataset/DAWN)",
    )

    p.add_argument(
        "--allow_fallback",
        action="store_true",
        help="If YOLO26 can't be loaded, allow fallback to a closest Ultralytics model.",
    )
    p.add_argument(
        "--fallback_model",
        default="yolo26n.pt",
        help="Fallback model to use when --allow_fallback is enabled.",
    )

    return p.parse_args()


def _ensure_dataset(raw_root: Path, processed_root: Path, *, seed: int) -> None:
    # Ensure processed dataset exists AND matches requested class space.
    imgs = processed_root / "images" / "train"
    lbls = processed_root / "labels" / "train"
    meta = processed_root / "_meta.json"
    if imgs.exists() and lbls.exists() and meta.exists():
        try:
            import json

            m = json.loads(meta.read_text(encoding="utf-8"))
            if m.get("classes") == list(BASELINE_CLASSES):
                return
        except Exception:
            pass

    print(f"[DAWN] Preparing YOLO dataset at: {processed_root}")
    prepare_dawn_yolo(
        raw_root=raw_root,
        out_root=processed_root,
        seed=int(seed),
        ratios=(0.7, 0.15, 0.15),
        classes=BASELINE_CLASSES,
    )


def main() -> None:
    args = _parse_args()
    set_seed(int(args.seed))

    raw_root = dawn_raw_root(args.raw_root)
    processed_root = dawn_processed_root()

    _ensure_dataset(raw_root=raw_root, processed_root=processed_root, seed=int(args.seed))

    # Verify data.yaml exists
    data_cfg = load_data_yaml(args.data)
    # Ultralytics expects dataset at data_cfg['path']
    # If user kept default configs/dawn.yaml, this points to data/processed/dawn_yolo.
    cfg_path = data_cfg.get("path")
    if cfg_path is None:
        raise ValueError("Dataset yaml must contain 'path'")

    # Model load: strict behavior
    res = try_load_ultralytics_model(args.model, allow_fallback=bool(args.allow_fallback), fallback_model=str(args.fallback_model))
    print(res.message)
    if not res.ok or res.model is None:
        raise SystemExit(2)

    model = res.model

    # Train
    train_kwargs = dict(
        data=str(args.data),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=str(args.name),
        seed=int(args.seed),
    )

    print("[YOLO] Training config:")
    for k, v in train_kwargs.items():
        print(f"  - {k}: {v}")

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
