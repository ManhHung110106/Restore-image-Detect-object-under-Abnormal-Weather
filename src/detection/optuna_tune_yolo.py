import os
import glob
import argparse
from pathlib import Path

import cv2
import optuna
import numpy as np
import torch

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
except ImportError:
    print("Warning: torchmetrics not found. Please install: pip install torchmetrics")
    MeanAveragePrecision = None

# Import our dehaze module
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.filters.dark_channel_prior import dehaze
from src.utils.yolo26 import try_load_ultralytics_model

def read_yolo_labels(label_path, img_w, img_h):
    boxes = []
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:5])
                    # unnormalize
                    x1 = (x_c - w/2) * img_w
                    y1 = (y_c - h/2) * img_h
                    x2 = (x_c + w/2) * img_w
                    y2 = (y_c + h/2) * img_h
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls_id)
    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

def objective(trial, image_paths, label_dirs, yolo_model, max_images=100):
    """
    Optuna objective function to maximize YOLO mAP and Recall on DAWN training/val set.
    """
    omega = trial.suggest_float("omega", 0.7, 0.99)
    t0 = trial.suggest_float("t0", 0.05, 0.2)
    patch_size = trial.suggest_int("patch_size", 5, 25, step=2)
    
    # Sub-sample image_paths for speed
    random_indices = np.random.choice(len(image_paths), min(max_images, len(image_paths)), replace=False)
    
    preds = []
    targets = []
    
    for i in random_indices:
        impath, split_type = image_paths[i]
        hazy = cv2.imread(impath)
        if hazy is None: continue
        h, w = hazy.shape[:2]
        
        # Dehaze
        dehazed = dehaze(hazy, patch_size=patch_size, omega=omega, t0=t0, refine=True)
        
        # YOLO inference
        results = yolo_model(dehazed, verbose=False)[0]
        pred_boxes_raw = results.boxes.xyxy.cpu()
        pred_scores_raw = results.boxes.conf.cpu()
        pred_labels_raw = results.boxes.cls.cpu().to(torch.int64)
        
        mapped_boxes = []
        mapped_scores = []
        mapped_labels = []
        for b, s, l in zip(pred_boxes_raw, pred_scores_raw, pred_labels_raw):
            l = int(l)
            if l == 0:
                mapped_boxes.append(b)
                mapped_scores.append(s)
                mapped_labels.append(1)
            elif l in [2, 3, 5, 7]:
                mapped_boxes.append(b)
                mapped_scores.append(s)
                mapped_labels.append(0)
                
        if len(mapped_boxes) > 0:
            pred_boxes = torch.stack(mapped_boxes)
            pred_scores = torch.tensor(mapped_scores)
            pred_labels = torch.tensor(mapped_labels, dtype=torch.int64)
        else:
            pred_boxes = torch.empty((0,4))
            pred_scores = torch.empty((0,))
            pred_labels = torch.empty((0,), dtype=torch.int64)
        
        # Ground truth
        # format: data/processed/dawn_yolo/images/train/Fog/001.jpg
        # label : data/processed/dawn_yolo/labels/train/Fog/001.txt
        img_name = os.path.basename(impath)
        cat_name = os.path.basename(os.path.dirname(impath))
        
        lbl_dir = label_dirs[split_type]
        lbl_path = os.path.join(lbl_dir, cat_name, os.path.splitext(img_name)[0] + ".txt")
        if not os.path.exists(lbl_path):
            # Sometimes it's directly in the split folder without weather category
            lbl_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + ".txt")
            
        gt_boxes_np, gt_labels_np = read_yolo_labels(lbl_path, w, h)
        gt_boxes = torch.from_numpy(gt_boxes_np)
        gt_labels = torch.from_numpy(gt_labels_np)
        
        if len(gt_boxes) > 0:
            preds.append(dict(boxes=pred_boxes, scores=pred_scores, labels=pred_labels))
            targets.append(dict(boxes=gt_boxes, labels=gt_labels))
        else:
            preds.append(dict(boxes=pred_boxes, scores=pred_scores, labels=pred_labels))
            targets.append(dict(boxes=torch.empty((0,4)), labels=torch.empty((0,), dtype=torch.int64)))
            
    if not preds or MeanAveragePrecision is None:
        return 0.0, 0.0, 0.0
        
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    metric.update(preds, targets)
    m = metric.compute()
    
    map_50_95 = m["map"].item()
    map_50 = m["map_50"].item()
    recall = m["mar_100"].item() # Using mar_100 as a proxy for recall
    
    # We want to maximize all three
    return map_50_95, map_50, recall

def run_optuna_tuning(dawn_dir, n_trials=50, out_db="sqlite:///optuna_yolo.db", max_images=100):
    # Load model
    load_res = try_load_ultralytics_model("yolo26n.pt", allow_fallback=True)
    if not load_res.ok:
        print("Cannot load YOLO model. Exiting.")
        return
    yolo_model = load_res.model
    
    # Get image paths
    image_paths = []
    label_dirs = {}
    
    for split in ["train", "val"]:
        fog_dir = os.path.join(dawn_dir, "images", split, "Fog")
        label_dirs[split] = os.path.join(dawn_dir, "labels", split)
        if os.path.exists(fog_dir):
            for ext in ["*.jpg", "*.png", "*.jpeg"]:
                for p in glob.glob(os.path.join(fog_dir, "**", ext), recursive=True):
                    image_paths.append((p, split))
                    
    print(f"Found {len(image_paths)} images for tuning.")
    
    study = optuna.create_study(
        study_name="dcp_yolo_optimization",
        storage=out_db,
        load_if_exists=True,
        directions=["maximize", "maximize", "maximize"]
    )
    
    study.optimize(lambda trial: objective(trial, image_paths, label_dirs, yolo_model, max_images=max_images), n_trials=n_trials)
    
    print("Pareto front trials:")
    for t in study.best_trials:
        print(f"  Trial#{t.number} Values (mAP@0.5:0.95, mAP@0.5, Recall): {t.values} Params: {t.params}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dawn_dir", default="data/processed/dawn_yolo")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--out_db", default="sqlite:///optuna_yolo.db", help="Optuna database URI")
    parser.add_argument("--max_images", type=int, default=100, help="Max images per trial")
    args = parser.parse_args()
    
    run_optuna_tuning(args.dawn_dir, n_trials=args.trials, out_db=args.out_db, max_images=args.max_images)
