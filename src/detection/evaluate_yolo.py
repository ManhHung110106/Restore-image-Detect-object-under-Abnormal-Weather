"""Evaluate YOLO (prefer YOLO26) baseline on DAWN (no preprocessing).

Outputs (default):
  results/baseline/
    metrics.json
    metrics.csv
    per_class_metrics.csv
    per_weather_metrics.csv
    confusion_matrix.png
    predictions_visualized/
    latency.json

Key metrics:
- mAP@0.5
- mAP@0.5:0.95
- precision/recall/F1 at configurable thresholds
- mean matched IoU at configurable thresholds
- per-class AP (from torchmetrics)
- per-weather metrics
- confusion matrix (detection-style, includes background row/col)
- inference latency + FPS

Notes:
- Ensures YOLO-formatted DAWN dataset exists under data/processed/dawn_yolo.
- Uses the test split by default.
"""

from __future__ import annotations

import sys
import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow running as a script: `python src/detection/evaluate_yolo.py ...`
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pandas as pd

from src.data.dawn_prepare import BASELINE_CLASSES, WEATHER_FOLDERS, prepare_dawn_yolo
from src.utils.io import write_csv, write_json
from src.utils.metrics import (
    MatchStats,
    greedy_match,
    mean_iou,
    merge_stats,
    per_class_stats_from_matches,
    precision_recall_f1,
    update_confusion_matrix,
)
from src.utils.paths import dawn_processed_root, dawn_raw_root
from src.utils.seed import set_seed
from src.utils.ultralytics_data import class_names_from_data, load_data_yaml
from src.utils.visualize import draw_boxes_bgr, save_image
from src.utils.yolo26 import try_load_ultralytics_model


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--data", required=True, help="Ultralytics data.yaml (e.g., configs/dawn.yaml)")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--weights", help="Weights path (e.g., runs/.../best.pt)")
    g.add_argument("--model", help="Model spec (e.g., yolo26n.pt)")

    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="auto")

    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for PR/F1 and matching")
    p.add_argument("--iou_nms", type=float, default=0.7, help="NMS IoU threshold for Ultralytics predict")
    p.add_argument("--iou_match", type=float, default=0.5, help="IoU threshold for greedy matching")

    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out_dir", default="results/baseline")

    p.add_argument(
        "--weather",
        default="all",
        choices=["all", "Fog", "Rain", "Snow", "Sand"],
        help="Evaluate only a specific weather subset (default: all)",
    )

    p.add_argument(
        "--raw_root",
        default=None,
        help="Raw DAWN dataset root (default: env DAWN_RAW_ROOT or ./dataset/DAWN)",
    )

    p.add_argument(
        "--processed_root",
        default=None,
        help="Override processed YOLO dataset root (folder with images/ and labels/). If set, dataset will NOT be auto-prepared.",
    )

    p.add_argument("--seed", type=int, default=42)

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

    p.add_argument("--viz_count", type=int, default=50, help="Number of prediction visualizations to save")
    p.add_argument("--max_images", type=int, default=0, help="Limit evaluation images (0 = all)")

    return p.parse_args()


def _resolve_processed_root(args: argparse.Namespace, data_cfg: Dict[str, Any]) -> Path:
    # Priority:
    # 1) explicit CLI --processed_root
    # 2) data.yaml 'path'
    # 3) default dawn_processed_root()
    if getattr(args, "processed_root", None):
        return Path(str(args.processed_root)).expanduser().resolve()

    cfg_path = data_cfg.get("path")
    if isinstance(cfg_path, str) and cfg_path.strip():
        p = Path(cfg_path)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        return p

    return dawn_processed_root()


def _ensure_dataset(raw_root: Path, processed_root: Path, *, seed: int) -> None:
    imgs = processed_root / "images" / "train"
    lbls = processed_root / "labels" / "train"
    meta = processed_root / "_meta.json"
    if imgs.exists() and lbls.exists() and meta.exists():
        try:
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


def _list_split_images(processed_root: Path, split: str) -> List[Path]:
    root = processed_root / "images" / split
    if not root.exists():
        return []
    imgs = []
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        imgs.extend(root.rglob(ext))
    return sorted(imgs)


def _weather_from_path(p: Path) -> str:
    # expects .../images/<split>/<Weather>/file.jpg
    for w in WEATHER_FOLDERS:
        if w in p.parts:
            return w
    return "Unknown"


def _label_path_for_image(processed_root: Path, split: str, img_path: Path) -> Path:
    # Map: images/<split>/... -> labels/<split>/...
    parts = list(img_path.parts)
    # Find 'images' segment
    try:
        idx = parts.index("images")
    except ValueError:
        raise ValueError(f"Unexpected image path (missing 'images'): {img_path}")

    parts[idx] = "labels"
    lbl = Path(*parts)
    return lbl.with_suffix(".txt")


def _read_yolo_label(label_path: Path, *, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    boxes: List[List[float]] = []
    labels: List[int] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc = float(parts[1]) * img_w
        yc = float(parts[2]) * img_h
        bw = float(parts[3]) * img_w
        bh = float(parts[4]) * img_h
        x1 = xc - bw / 2.0
        y1 = yc - bh / 2.0
        x2 = xc + bw / 2.0
        y2 = yc + bh / 2.0
        boxes.append([x1, y1, x2, y2])
        labels.append(cls)

    return np.asarray(boxes, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def _device_arg(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _get_model_names(model: Any) -> Optional[Any]:
    names = getattr(model, "names", None)
    if names is not None:
        return names
    inner = getattr(model, "model", None)
    return getattr(inner, "names", None)


def _name_for_class_id(names: Any, class_id: int) -> Optional[str]:
    try:
        if isinstance(names, dict):
            v = names.get(class_id)
            return str(v) if v is not None else None
        if isinstance(names, (list, tuple)):
            if 0 <= class_id < len(names):
                return str(names[class_id])
            return None
    except Exception:
        return None
    return None


def _map_pred_labels_to_dataset(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_labels: np.ndarray,
    *,
    model_names: Any,
    class_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map model output labels (often COCO) into dataset label space using class names."""

    class_to_id = {n.lower(): i for i, n in enumerate(class_names)}
    target_set = set(class_to_id.keys())

    def map_name(label_name: str) -> Optional[str]:
        raw = str(label_name).strip().lower()

        # If dataset wants vehicle/person, map common transport labels to vehicle.
        if "vehicle" in target_set and raw in {"car", "bus", "train", "motorcycle", "motorbike", "truck", "van"}:
            return "vehicle"
        if "person" in target_set and raw in {"person", "people", "pedestrian"}:
            return "person"

        # If model already emits target labels, keep them.
        if raw in target_set:
            return raw

        return None

    keep: List[int] = []
    mapped: List[int] = []

    for j, lbl in enumerate(pred_labels.tolist() if hasattr(pred_labels, "tolist") else list(pred_labels)):
        name = _name_for_class_id(model_names, int(lbl))
        if not name:
            continue
        m = map_name(name)
        if m is None:
            continue
        keep.append(j)
        mapped.append(class_to_id[m])

    if not keep:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    pred_boxes = pred_boxes[np.asarray(keep, dtype=np.int64)]
    pred_scores = pred_scores[np.asarray(keep, dtype=np.int64)]
    pred_labels = np.asarray(mapped, dtype=np.int64)
    return pred_boxes, pred_scores, pred_labels


@dataclass
class LatencyStats:
    n: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    fps_mean: float


def _latency_summary(times_s: List[float]) -> LatencyStats:
    arr = np.asarray(times_s, dtype=np.float64)
    if arr.size == 0:
        return LatencyStats(n=0, mean_ms=0.0, median_ms=0.0, p95_ms=0.0, fps_mean=0.0)

    mean_ms = float(arr.mean() * 1000.0)
    median_ms = float(np.median(arr) * 1000.0)
    p95_ms = float(np.percentile(arr, 95) * 1000.0)
    fps_mean = float(1000.0 / mean_ms) if mean_ms > 1e-9 else 0.0
    return LatencyStats(n=int(arr.size), mean_ms=mean_ms, median_ms=median_ms, p95_ms=p95_ms, fps_mean=fps_mean)


def main() -> None:
    args = _parse_args()
    set_seed(int(args.seed))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_root = dawn_raw_root(args.raw_root)
    data = load_data_yaml(args.data)
    processed_root = _resolve_processed_root(args, data)

    # Prepare only when using the default processed dataset root.
    # If user overrides processed_root (e.g. DCP-preprocessed images), we assume it's already prepared.
    if processed_root.resolve() == dawn_processed_root().resolve() and args.processed_root is None:
        _ensure_dataset(raw_root=raw_root, processed_root=processed_root, seed=int(args.seed))
    else:
        if not (processed_root / "images").exists():
            raise FileNotFoundError(f"Processed dataset missing images/: {processed_root}")
        if not (processed_root / "labels").exists():
            raise FileNotFoundError(f"Processed dataset missing labels/: {processed_root}")

    class_names = class_names_from_data(data)
    nc = len(class_names)

    model_spec = args.weights if args.weights else args.model

    res = try_load_ultralytics_model(model_spec, allow_fallback=bool(args.allow_fallback), fallback_model=str(args.fallback_model))
    print(res.message)
    if not res.ok or res.model is None:
        raise SystemExit(2)

    model = res.model
    model_names = _get_model_names(model)

    # torchmetrics
    import torch
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    metric_50 = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5], class_metrics=True)
    metric_5095 = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

    per_weather_metric_50: Dict[str, MeanAveragePrecision] = {
        w: MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5], class_metrics=True) for w in WEATHER_FOLDERS
    }
    per_weather_metric_5095: Dict[str, MeanAveragePrecision] = {
        w: MeanAveragePrecision(iou_type="bbox", class_metrics=True) for w in WEATHER_FOLDERS
    }

    images = _list_split_images(processed_root, args.split)
    if not images:
        raise FileNotFoundError(f"No images found for split={args.split} in {processed_root}")

    if str(args.weather) != "all":
        images = [p for p in images if _weather_from_path(p) == str(args.weather)]
        if not images:
            raise FileNotFoundError(f"No images found for split={args.split} weather={args.weather} in {processed_root}")

    if int(args.max_images) and int(args.max_images) > 0:
        images = images[: int(args.max_images)]

    # Confusion matrix includes background
    bg = nc
    cm = np.zeros((nc + 1, nc + 1), dtype=np.int64)

    per_image_rows: List[Dict[str, Any]] = []

    stats_all: List[MatchStats] = []
    stats_by_weather: Dict[str, List[MatchStats]] = {w: [] for w in WEATHER_FOLDERS}

    per_class_agg = {i: MatchStats() for i in range(nc)}
    per_class_by_weather = {w: {i: MatchStats() for i in range(nc)} for w in WEATHER_FOLDERS}

    latency_times: List[float] = []
    latency_by_weather: Dict[str, List[float]] = {w: [] for w in WEATHER_FOLDERS}

    device_arg = _device_arg(str(args.device))

    # Warm-up (affects GPU timing)
    try:
        _ = model.predict(source=str(images[0]), imgsz=int(args.imgsz), conf=float(args.conf), iou=float(args.iou_nms), device=device_arg, verbose=False)
    except Exception:
        pass

    # Visualization
    # Prefer saving images that actually have predictions (above --conf), so users can
    # visually verify correctness.
    # Colors (BGR): ground-truth=red, predictions=green.
    # Thickness: 1 (thinner than the previous default).
    viz_dir = out_dir / "predictions_visualized"
    if int(args.viz_count) > 0 and viz_dir.exists():
        shutil.rmtree(viz_dir, ignore_errors=True)
    viz_saved = 0

    for idx, img_path in enumerate(images):
        weather = _weather_from_path(img_path)

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        lbl_path = _label_path_for_image(processed_root, args.split, img_path)
        gt_boxes, gt_labels = _read_yolo_label(lbl_path, img_w=w, img_h=h)

        # Guard against unexpected label ids in dataset
        if gt_labels.size:
            gt_mask = (gt_labels >= 0) & (gt_labels < nc)
            gt_boxes = gt_boxes[gt_mask]
            gt_labels = gt_labels[gt_mask]

        # Predict + latency
        t0 = time.perf_counter()
        results = model.predict(
            source=str(img_path),
            imgsz=int(args.imgsz),
            conf=float(args.conf),
            iou=float(args.iou_nms),
            device=device_arg,
            verbose=False,
        )
        dt = time.perf_counter() - t0
        latency_times.append(dt)
        if weather in latency_by_weather:
            latency_by_weather[weather].append(dt)

        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            pred_boxes = np.zeros((0, 4), dtype=np.float32)
            pred_scores = np.zeros((0,), dtype=np.float32)
            pred_labels = np.zeros((0,), dtype=np.int64)
        else:
            pred_boxes = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            pred_scores = boxes.conf.detach().cpu().numpy().astype(np.float32)
            pred_labels = boxes.cls.detach().cpu().numpy().astype(np.int64)

        # Map model label space (often COCO) into the dataset label space using model.names.
        # This makes evaluation/inference meaningful even when using pretrained weights.
        if pred_labels.size and model_names is not None:
            pred_boxes, pred_scores, pred_labels = _map_pred_labels_to_dataset(
                pred_boxes,
                pred_scores,
                pred_labels,
                model_names=model_names,
                class_names=class_names,
            )
        elif pred_labels.size:
            # Fallback: keep only in-range ids.
            pred_mask = (pred_labels >= 0) & (pred_labels < nc)
            pred_boxes = pred_boxes[pred_mask]
            pred_scores = pred_scores[pred_mask]
            pred_labels = pred_labels[pred_mask]

        # torchmetrics update
        pred_t = {
            "boxes": torch.tensor(pred_boxes, dtype=torch.float32),
            "scores": torch.tensor(pred_scores, dtype=torch.float32),
            "labels": torch.tensor(pred_labels, dtype=torch.int64),
        }
        tgt_t = {
            "boxes": torch.tensor(gt_boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_labels, dtype=torch.int64),
        }

        metric_50.update([pred_t], [tgt_t])
        metric_5095.update([pred_t], [tgt_t])

        if weather in per_weather_metric_50:
            per_weather_metric_50[weather].update([pred_t], [tgt_t])
            per_weather_metric_5095[weather].update([pred_t], [tgt_t])

        # Greedy matching for PR/F1, IoU, confusion
        stats, matches, unmatched_gt, unmatched_pred = greedy_match(
            pred_boxes,
            pred_labels,
            pred_scores,
            gt_boxes,
            gt_labels,
            iou_thres=float(args.iou_match),
            conf_thres=float(args.conf),
        )
        stats_all.append(stats)
        if weather in stats_by_weather:
            stats_by_weather[weather].append(stats)

        update_confusion_matrix(
            cm,
            pred_labels=pred_labels,
            pred_scores=pred_scores,
            gt_labels=gt_labels,
            matches=matches,
            unmatched_gt=unmatched_gt,
            unmatched_pred=unmatched_pred,
            conf_thres=float(args.conf),
            background_index=bg,
        )

        pc = per_class_stats_from_matches(
            nc=nc,
            gt_labels=gt_labels,
            pred_labels=pred_labels,
            pred_scores=pred_scores,
            matches=matches,
            unmatched_gt=unmatched_gt,
            unmatched_pred=unmatched_pred,
            conf_thres=float(args.conf),
        )
        for c in range(nc):
            per_class_agg[c].tp += pc[c].tp
            per_class_agg[c].fp += pc[c].fp
            per_class_agg[c].fn += pc[c].fn
            per_class_agg[c].iou_sum += pc[c].iou_sum
            per_class_agg[c].iou_count += pc[c].iou_count

            if weather in per_class_by_weather:
                per_class_by_weather[weather][c].tp += pc[c].tp
                per_class_by_weather[weather][c].fp += pc[c].fp
                per_class_by_weather[weather][c].fn += pc[c].fn
                per_class_by_weather[weather][c].iou_sum += pc[c].iou_sum
                per_class_by_weather[weather][c].iou_count += pc[c].iou_count

        # per-image row
        p, r, f1 = precision_recall_f1(stats.tp, stats.fp, stats.fn)
        per_image_rows.append(
            {
                "image": img_path.as_posix(),
                "weather": weather,
                "tp": stats.tp,
                "fp": stats.fp,
                "fn": stats.fn,
                "precision": p,
                "recall": r,
                "f1": f1,
                "mean_iou": mean_iou(stats),
                "latency_ms": dt * 1000.0,
            }
        )

        # Visualize
        if int(args.viz_count) > 0 and viz_saved < int(args.viz_count):
            # Show only predictions that meet the same threshold used for PR/F1 & matching.
            if pred_scores.size:
                viz_mask = pred_scores >= float(args.conf)
                pred_boxes_v = pred_boxes[viz_mask]
                pred_scores_v = pred_scores[viz_mask]
                pred_labels_v = pred_labels[viz_mask]
            else:
                pred_boxes_v = pred_boxes
                pred_scores_v = pred_scores
                pred_labels_v = pred_labels

            # If this image has no predictions above threshold, skip it so we don't
            # produce "GT-only" visuals that look like predictions are missing.
            if int(pred_boxes_v.shape[0]) > 0:
                overlay = draw_boxes_bgr(
                    img,
                    gt_boxes,
                    gt_labels,
                    None,
                    class_names,
                    color=(0, 0, 255),
                    thickness=1,
                )
                overlay = draw_boxes_bgr(
                    overlay,
                    pred_boxes_v,
                    pred_labels_v,
                    pred_scores_v,
                    class_names,
                    color=(0, 255, 0),
                    thickness=1,
                )
                save_image(viz_dir / weather / img_path.name, overlay)
                viz_saved += 1

        if (idx + 1) % 50 == 0 or (idx + 1) == len(images):
            print(f"Processed {idx + 1}/{len(images)}")

    # Compute torchmetrics outputs
    out_50 = metric_50.compute()
    out_5095 = metric_5095.compute()

    # Summaries for PR/F1/IoU
    merged = merge_stats(stats_all)
    prec, rec, f1 = precision_recall_f1(merged.tp, merged.fp, merged.fn)
    miou = mean_iou(merged)

    # Per-class AP
    def _to_list(x: Any) -> List[float]:
        if x is None:
            return []
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return [float(v) for v in np.asarray(x).reshape(-1).tolist()]

    ap50_per_class = _to_list(out_50.get("map_per_class"))
    ap5095_per_class = _to_list(out_5095.get("map_per_class"))

    per_class_rows: List[Dict[str, Any]] = []
    for i, name in enumerate(class_names):
        s = per_class_agg[i]
        p_i, r_i, f1_i = precision_recall_f1(s.tp, s.fp, s.fn)
        per_class_rows.append(
            {
                "class_id": i,
                "class": name,
                "ap50": ap50_per_class[i] if i < len(ap50_per_class) else None,
                "ap5095": ap5095_per_class[i] if i < len(ap5095_per_class) else None,
                "precision": p_i,
                "recall": r_i,
                "f1": f1_i,
                "mean_iou": mean_iou(s),
                "tp": s.tp,
                "fp": s.fp,
                "fn": s.fn,
            }
        )

    # Per-weather summary
    per_weather_rows: List[Dict[str, Any]] = []
    for wthr in WEATHER_FOLDERS:
        s = merge_stats(stats_by_weather[wthr])
        p_w, r_w, f1_w = precision_recall_f1(s.tp, s.fp, s.fn)
        miou_w = mean_iou(s)

        if len(stats_by_weather[wthr]) > 0:
            o50 = per_weather_metric_50[wthr].compute()
            o5095 = per_weather_metric_5095[wthr].compute()
            map50 = float(o50.get("map", 0.0))
            map5095 = float(o5095.get("map", 0.0))
        else:
            map50 = 0.0
            map5095 = 0.0

        per_weather_rows.append(
            {
                "weather": wthr,
                "n_images": len(stats_by_weather[wthr]),
                "map50": map50,
                "map5095": map5095,
                "precision": p_w,
                "recall": r_w,
                "f1": f1_w,
                "mean_iou": miou_w,
            }
        )

    # Latency summary
    lat_all = _latency_summary(latency_times)
    lat_by_weather = {w: asdict(_latency_summary(latency_by_weather[w])) for w in WEATHER_FOLDERS}

    # Confusion matrix plot
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        labels = class_names + ["background"]
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("Confusion Matrix (Detection-style)")
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png", dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] Failed to plot confusion matrix: {e}")

    # Export CSVs
    per_image_df = pd.DataFrame(per_image_rows)
    write_csv(out_dir / "per_image.csv", per_image_df)

    per_class_df = pd.DataFrame(per_class_rows)
    write_csv(out_dir / "per_class_metrics.csv", per_class_df)

    per_weather_df = pd.DataFrame(per_weather_rows)
    write_csv(out_dir / "per_weather_metrics.csv", per_weather_df)

    summary = {
        "split": args.split,
        "n_images": len(images),
        "model": str(model_spec),
        "imgsz": int(args.imgsz),
        "conf": float(args.conf),
        "iou_nms": float(args.iou_nms),
        "iou_match": float(args.iou_match),
        "map50": float(out_50.get("map", 0.0)),
        "map5095": float(out_5095.get("map", 0.0)),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mean_iou": miou,
        "latency": asdict(lat_all),
        "latency_by_weather": lat_by_weather,
    }

    write_json(out_dir / "metrics.json", summary)
    write_json(out_dir / "latency.json", {"overall": asdict(lat_all), "by_weather": lat_by_weather})

    # metrics.csv as 1-row table
    write_csv(out_dir / "metrics.csv", pd.DataFrame([summary]))

    print("\n[Done] Wrote results to:", out_dir.resolve())


if __name__ == "__main__":
    main()

'''
python src\detection\evaluate_yolo.py `
  --data configs\dawn.yaml `
  --model yolo26n.pt `
  --imgsz 640 `
  --device auto `
  --out_dir results\baseline `
  --seed 42 `
  --viz_count 20 `
  --max_images 20
'''
