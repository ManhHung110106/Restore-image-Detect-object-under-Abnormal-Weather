import os
import yaml
import pandas as pd
import numpy as np
import cv2
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.restoration.wmgf_derain import WMGFConfig, WMGFDerainFilter
from src.datasets.rain100H_dataset import PairedRainDataset
from src.datasets.dawn_dataset import DawnDataset
from src.metrics.full_reference import compute_ssim, compute_psnr
from src.metrics.no_reference import compute_brisque, compute_niqe, compute_piqe, compute_entropy
from src.detection.yolo_runner import YOLOEvaluator
from src.metrics.detection import compute_mean_iou
from ultralytics import YOLO

def load_config(path, is_optuna=False):
    if not os.path.exists(path):
        print(f"Warning: config not found {path}")
        return None
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if is_optuna:
        return WMGFConfig(**cfg)
    return WMGFConfig(**cfg)

def evaluate_on_paired(filter_wmgf, dataset_path):
    if not os.path.exists(dataset_path):
        return None, None
    dataset = PairedRainDataset(dataset_path, "A:/HUST_on_GitHub/ProjectCV/dataset/rain100H")
    ssims, psnrs = [], []
    for sample in dataset:
        restored = filter_wmgf.restore(sample["rainy"])
        ssims.append(compute_ssim(restored, sample["clean"]))
        psnrs.append(compute_psnr(restored, sample["clean"]))
    return np.mean(ssims), np.mean(psnrs)

def evaluate_on_dawn_nr(filter_wmgf, dataset_path):
    if not os.path.exists(dataset_path):
        return None, None, None, None
    dataset = DawnDataset(dataset_path, "")
    brisques, niqes, piqes, entropies = [], [], [], []
    for sample in dataset:
        restored = filter_wmgf.restore(sample["image"])
        b = compute_brisque(restored)
        n = compute_niqe(restored)
        p = compute_piqe(restored)
        e = compute_entropy(restored)
        
        if not np.isnan(b): brisques.append(b)
        if not np.isnan(n): niqes.append(n)
        if not np.isnan(p): piqes.append(p)
        if not np.isnan(e): entropies.append(e)
        
    return np.mean(brisques), np.mean(niqes), np.mean(piqes), np.mean(entropies)

def evaluate_on_dawn_yolo(filter_wmgf, dataset_path, results_dir):
    if not os.path.exists(dataset_path):
        return None
    
    dataset = DawnDataset(dataset_path, "")
    os.makedirs(results_dir, exist_ok=True)
    
    for sample in dataset:
        restored = filter_wmgf.restore(sample["image"])
        basename = sample["image_rel"]
        cv2.imwrite(os.path.join(results_dir, basename), restored)
        
        if sample.get("label_path") and os.path.exists(sample["label_path"]):
            import shutil
            labels_dir = os.path.join(os.path.dirname(results_dir), "labels")
            os.makedirs(labels_dir, exist_ok=True)
            label_out_path = os.path.join(labels_dir, sample["label_rel"])
            shutil.copy2(sample["label_path"], label_out_path)
        
    temp_yaml_path = os.path.join(results_dir, "temp_eval.yaml")
    lines = [
        f"path: A:/HUST_on_GitHub/ProjectCV/data/dawn_rain",
        f"train: A:/HUST_on_GitHub/ProjectCV/data/dawn_rain/images",
        f"val: {results_dir}", 
        f"names:",
        f"  0: person",
        f"  1: bicycle",
        f"  2: car",
        f"  3: motorcycle",
        f"  4: airplane",
        f"  5: bus",
        f"  6: train",
        f"  7: truck"
    ]
    with open(temp_yaml_path, "w") as f:
        f.write("\n".join(lines))
        
    evaluator = YOLOEvaluator("A:/HUST_on_GitHub/ProjectCV/yolo26n.pt", temp_yaml_path)
    metrics = evaluator.evaluate(temp_yaml_path, classes=[0, 1, 2, 3, 5, 7])
    
    yolo_model = YOLO("A:/HUST_on_GitHub/ProjectCV/yolo26n.pt")
    labels_dir = os.path.join(os.path.dirname(results_dir), "labels")
    mean_iou = compute_mean_iou(yolo_model, results_dir, labels_dir)
    
    if metrics is None:
        metrics = {}
    metrics['mean_iou'] = mean_iou
    
    return metrics

def main():
    configs = {
        "Raw": None,
        "Default": load_config("A:/HUST_on_GitHub/ProjectCV/configs/wmgf/derain_wmgf_default.yaml"),
        "Config1": load_config("A:/HUST_on_GitHub/ProjectCV/configs/wmgf/derain_optuna_config1_ssim.yaml", True),
        "Config2": load_config("A:/HUST_on_GitHub/ProjectCV/configs/wmgf/derain_optuna_config2_brisque.yaml", True),
        "Config3": load_config("A:/HUST_on_GitHub/ProjectCV/configs/wmgf/derain_optuna_config3_map50.yaml", True)
    }
    
    test_file_paired = "A:/HUST_on_GitHub/ProjectCV/data/rain100h/splits/test.txt"
    dawn_test_file = "A:/HUST_on_GitHub/ProjectCV/data/dawn_rain/splits/rain_test_pairs.csv"
    
    results = []
    
    for name, cfg in configs.items():
            
        print(f"Evaluating {name}...")
        wmgf = WMGFDerainFilter(cfg) if cfg is not None else None
        
        print(f"[{name}] Computing FR metrics on paired dataset...")
        ssim, psnr = evaluate_on_paired(wmgf, test_file_paired) if wmgf else evaluate_on_paired_raw(test_file_paired)
        
        print(f"[{name}] Computing NR metrics on DAWN rain...")
        brisque, niqe, piqe, entropy = evaluate_on_dawn_nr(wmgf, dawn_test_file) if wmgf else evaluate_on_dawn_nr_raw(dawn_test_file)
        
        print(f"[{name}] Computing YOLO detection metrics on DAWN rain...")
        yolo_dir = f"A:/HUST_on_GitHub/ProjectCV/paper/figures/wmgf_detection/{name}"
        os.makedirs(yolo_dir, exist_ok=True)
        
        yolo_metrics = evaluate_on_dawn_yolo(wmgf, dawn_test_file, yolo_dir) if wmgf else evaluate_on_dawn_yolo_raw(dawn_test_file, yolo_dir)
        
        results.append({
            "Filter": "WMGF",
            "Config": name,
            "Phase1_SSIM": ssim,
            "Phase1_PSNR": psnr,
            "Phase2_BRISQUE": brisque,
            "Phase2_NIQE": niqe,
            "Phase2_mAP50": yolo_metrics.get("map50", 0) if yolo_metrics else 0,
            "Phase2_mAP50-95": yolo_metrics.get("map50_95", 0) if yolo_metrics else 0,
            "Phase2_meanIoU": yolo_metrics.get("mean_iou", 0) if yolo_metrics else 0
        })
        
    df = pd.DataFrame(results)
    os.makedirs("A:/HUST_on_GitHub/ProjectCV/paper/tables", exist_ok=True)
    df.to_csv("A:/HUST_on_GitHub/ProjectCV/paper/tables/wmgf_evaluation.csv", index=False)
    print("\n--- FINAL WMGF EVALUATION RESULTS ---")
    print(df.to_string())

def evaluate_on_paired_raw(dataset_path):
    if not os.path.exists(dataset_path): return 0, 0
    dataset = PairedRainDataset(dataset_path, "A:/HUST_on_GitHub/ProjectCV/dataset/rain100H")
    ssims, psnrs = [], []
    for sample in dataset:
        ssims.append(compute_ssim(sample["rainy"], sample["clean"]))
        psnrs.append(compute_psnr(sample["rainy"], sample["clean"]))
    return np.mean(ssims), np.mean(psnrs)

def evaluate_on_dawn_nr_raw(dataset_path):
    if not os.path.exists(dataset_path): return 0, 0, 0, 0
    dataset = DawnDataset(dataset_path, "")
    brisques, niqes, piqes, entropies = [], [], [], []
    for sample in dataset:
        b = compute_brisque(sample["image"])
        n = compute_niqe(sample["image"])
        p = compute_piqe(sample["image"])
        e = compute_entropy(sample["image"])
        if not np.isnan(b): brisques.append(b)
        if not np.isnan(n): niqes.append(n)
        if not np.isnan(p): piqes.append(p)
        if not np.isnan(e): entropies.append(e)
    return np.mean(brisques), np.mean(niqes), np.mean(piqes), np.mean(entropies)

def evaluate_on_dawn_yolo_raw(dataset_path, results_dir):
    if not os.path.exists(dataset_path): return None
    dataset = DawnDataset(dataset_path, "")
    import shutil
    for sample in dataset:
        basename = sample["image_rel"]
        shutil.copy2(sample["image_path"], os.path.join(results_dir, basename))
        if sample.get("label_path") and os.path.exists(sample["label_path"]):
            labels_dir = os.path.join(os.path.dirname(results_dir), "labels")
            os.makedirs(labels_dir, exist_ok=True)
            label_out_path = os.path.join(labels_dir, sample["label_rel"])
            shutil.copy2(sample["label_path"], label_out_path)
    
    temp_yaml_path = os.path.join(results_dir, "temp_eval.yaml")
    lines = [
        f"path: A:/HUST_on_GitHub/ProjectCV/data/dawn_rain",
        f"train: A:/HUST_on_GitHub/ProjectCV/data/dawn_rain/images",
        f"val: {results_dir}", 
        f"names:\n  0: person\n  1: bicycle\n  2: car\n  3: motorcycle\n  4: airplane\n  5: bus\n  6: train\n  7: truck"
    ]
    with open(temp_yaml_path, "w") as f: f.write("\n".join(lines))
    evaluator = YOLOEvaluator("A:/HUST_on_GitHub/ProjectCV/yolo26n.pt", temp_yaml_path)
    metrics = evaluator.evaluate(temp_yaml_path, classes=[0, 1, 2, 3, 5, 7])
    yolo_model = YOLO("A:/HUST_on_GitHub/ProjectCV/yolo26n.pt")
    labels_dir = os.path.join(os.path.dirname(results_dir), "labels")
    mean_iou = compute_mean_iou(yolo_model, results_dir, labels_dir)
    if metrics is None: metrics = {}
    metrics['mean_iou'] = mean_iou
    return metrics

if __name__ == "__main__":
    main()
