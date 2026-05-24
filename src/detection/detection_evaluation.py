"""
Detection evaluation utilities for comparing dehazing configs with YOLOv26
"""
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
except ImportError:
    MeanAveragePrecision = None


def draw_bounding_boxes(img, boxes, labels, color, thickness=2, label_names=None):
    """
    Draw bounding boxes on an image.
    
    Args:
        img: Input image (RGB format)
        boxes: Tensor of shape (N, 4) with boxes in xyxy format
        labels: Tensor of shape (N,) with class labels
        color: Tuple (R, G, B) for color
        thickness: Line thickness
        label_names: Dict mapping class ID to name
    
    Returns:
        Image with drawn bounding boxes
    """
    img_copy = img.copy()
    if len(boxes) == 0:
        return img_copy
    
    boxes_np = boxes.cpu().numpy() if hasattr(boxes, 'cpu') else boxes
    labels_np = labels.cpu().numpy() if hasattr(labels, 'cpu') else labels
    
    for box, label in zip(boxes_np, labels_np):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
        
        if label_names:
            label_name = label_names.get(int(label), f"Class {int(label)}")
        else:
            label_name = f"C{int(label)}"
        cv2.putText(img_copy, label_name, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return img_copy


def compute_map_metrics(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
    """
    Compute mAP@50, mAP@95, and Recall metrics
    
    Args:
        pred_boxes: Predicted boxes tensor (N, 4) in xyxy format
        pred_labels: Predicted labels tensor (N,)
        pred_scores: Predicted confidence scores tensor (N,)
        gt_boxes: Ground truth boxes tensor (M, 4) in xyxy format
        gt_labels: Ground truth labels tensor (M,)
    
    Returns:
        Dict with map50, map95, and mar_100 values
    """
    if MeanAveragePrecision is None or len(gt_boxes) == 0:
        return {"map50": 0.0, "map95": 0.0, "mar_100": 0.0}
    
    try:
        preds = [dict(boxes=pred_boxes, scores=pred_scores, labels=pred_labels)]
        targets = [dict(boxes=gt_boxes, labels=gt_labels)]
        
        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        metric.update(preds, targets)
        m = metric.compute()
        
        return {
            "map50": m["map_50"].item(),
            "map95": m["map"].item(),
            "mar_100": m["mar_100"].item()
        }
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {"map50": 0.0, "map95": 0.0, "mar_100": 0.0}


def load_dawn_samples(dawn_root: str = "data/processed/dawn_yolo", 
                      max_per_weather: int = 8) -> List[Tuple[str, str, str]]:
    """
    Load DAWN dataset samples from disk
    
    Args:
        dawn_root: Root directory of DAWN dataset
        max_per_weather: Maximum samples per weather type
    
    Returns:
        List of tuples (img_path, label_path, weather_type)
    """
    dawn_root = Path(dawn_root)
    samples = []
    
    for weather_type in ["Fog", "Rain", "Sand", "Snow"]:
        img_dir = dawn_root / f"images/train/{weather_type}"
        label_dir = dawn_root / f"labels/train/{weather_type}"
        
        if img_dir.exists() and label_dir.exists():
            img_files = sorted(list(img_dir.glob("*.jpg")))[:max_per_weather]
            for img_file in img_files:
                label_file = label_dir / img_file.stem
                label_path = label_file.with_suffix(".txt")
                if label_path.exists():
                    samples.append((str(img_file), str(label_path), weather_type))
    
    return samples


def map_yolo_to_simple_labels(yolo_labels):
    """
    Map YOLO classes to simple labels: Person (1) and Vehicle (0)
    
    YOLO: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
    
    Args:
        yolo_labels: Tensor of YOLO class labels
    
    Returns:
        Tensor of remapped labels (0=vehicle, 1=person)
    """
    mapped = []
    for l in yolo_labels:
        l_int = int(l)
        if l_int == 0:  # person -> 1
            mapped.append(1)
        elif l_int in [2, 3, 5, 7]:  # vehicle -> 0
            mapped.append(0)
    return mapped


def filter_predictions_by_class(boxes, labels, scores, yolo_classes):
    """
    Filter YOLO predictions by specific YOLO class IDs and map to simple labels
    
    Args:
        boxes: Prediction boxes (N, 4) in xyxy format
        labels: YOLO class labels (N,)
        scores: Confidence scores (N,)
        yolo_classes: List of YOLO class IDs to keep
    
    Returns:
        Tuple of (filtered_boxes, mapped_labels, filtered_scores)
    """
    filtered_boxes = []
    filtered_scores = []
    mapped_labels = []
    
    boxes_np = boxes.cpu().numpy() if hasattr(boxes, 'cpu') else boxes
    labels_np = labels.cpu().numpy() if hasattr(labels, 'cpu') else labels
    scores_np = scores.cpu().numpy() if hasattr(scores, 'cpu') else scores
    
    for b, l, s in zip(boxes_np, labels_np, scores_np):
        l_int = int(l)
        if l_int in yolo_classes:
            filtered_boxes.append(b)
            filtered_scores.append(s)
            # Map to simple labels
            if l_int == 0:  # person -> 1
                mapped_labels.append(1)
            else:  # vehicle -> 0
                mapped_labels.append(0)
    
    if filtered_boxes:
        filtered_boxes = torch.from_numpy(np.array(filtered_boxes)).float()
        filtered_scores = torch.tensor(filtered_scores)
        mapped_labels = torch.tensor(mapped_labels, dtype=torch.int64)
    else:
        filtered_boxes = torch.empty((0, 4))
        filtered_scores = torch.empty((0,))
        mapped_labels = torch.empty((0,), dtype=torch.int64)
    
    return filtered_boxes, mapped_labels, filtered_scores
