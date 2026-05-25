import os
import yaml
import json
import pandas as pd
import numpy as np
import cv2
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.restoration.dcp import DCPConfig, DCPDehazeFilter
from src.datasets.reside_dataset import ResideDataset
from src.datasets.dawn_dataset import DawnDataset
from src.metrics.full_reference import compute_ssim, compute_psnr
from src.metrics.no_reference import compute_brisque
from src.detection.yolo_runner import YOLOEvaluator

def load_config(path, is_optuna=False):
    if not os.path.exists(path):
        print(f"Warning: config not found {path}")
        return None
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if is_optuna:
        return DCPConfig(**cfg)
    return DCPConfig(
        patch_size=cfg.get("patch_size", 15),
        omega=cfg.get("omega", 0.95),
        t0=cfg.get("t0", 0.10),
        top_percent=cfg.get("top_percent", 0.001),
        guided_radius=cfg.get("guided_radius", 40),
        guided_eps=cfg.get("guided_eps", 0.001),
        gamma=cfg.get("gamma", 1.0),
        clahe_clip=cfg.get("clahe_clip", 0.0)
    )

def evaluate_on_reside(dcp, dataset_path):
    if not os.path.exists(dataset_path):
        return None, None
    dataset = ResideDataset(dataset_path, "A:/HUST_on_GitHub/ProjectCV/dataset/RESIDE-6K")
    ssims, psnrs = [], []
    for sample in dataset:
        restored = dcp.restore(sample["hazy"])
        ssims.append(compute_ssim(restored, sample["clear"]))
        psnrs.append(compute_psnr(restored, sample["clear"]))
    return np.mean(ssims), np.mean(psnrs)

def evaluate_on_dawn_nr(dcp, dataset_path):
    if not os.path.exists(dataset_path):
        return None
    dataset = DawnDataset(dataset_path, "")
    brisques = []
    for sample in dataset:
        restored = dcp.restore(sample["image"])
        brisques.append(compute_brisque(restored))
    return np.mean(brisques)

def evaluate_on_dawn_yolo(dcp, dataset_path, results_dir):
    if not os.path.exists(dataset_path):
        return None
    
    # Needs to save images
    dataset = DawnDataset(dataset_path, "")
    os.makedirs(results_dir, exist_ok=True)
    
    for sample in dataset:
        restored = dcp.restore(sample["image"])
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
        f"path: A:/HUST_on_GitHub/ProjectCV/data/dawn",
        f"train: A:/HUST_on_GitHub/ProjectCV/data/dawn/images",
        f"val: {results_dir}", 
        f"names:",
        f"  0: car",
        f"  1: bus",
        f"  2: truck",
        f"  3: motorcycle",
        f"  4: bicycle",
        f"  5: person"
    ]
    with open(temp_yaml_path, "w") as f:
        f.write("\n".join(lines))
        
    evaluator = YOLOEvaluator("A:/HUST_on_GitHub/ProjectCV/yolo26n.pt", temp_yaml_path)
    metrics = evaluator.evaluate(temp_yaml_path)
    return metrics["map50"]

def main():
    configs = {
        "Default": load_config("A:/HUST_on_GitHub/ProjectCV/configs/dcp_default.yaml"),
        "Optuna_SSIM": load_config("A:/HUST_on_GitHub/ProjectCV/results/optuna/config1_reside_ssim/best_config.yaml", True),
        "Optuna_BRISQUE": load_config("A:/HUST_on_GitHub/ProjectCV/results/optuna/config2_dawn_brisque/best_config.yaml", True),
        "Optuna_mAP50": load_config("A:/HUST_on_GitHub/ProjectCV/results/optuna/config3_dawn_map50/best_config.yaml", True)
    }
    
    reside_test_file = "A:/HUST_on_GitHub/ProjectCV/data/reside6k/splits/test.txt"
    dawn_test_file = "A:/HUST_on_GitHub/ProjectCV/data/dawn/splits/fog_test_pairs.csv"
    
    results = []
    
    for name, cfg in configs.items():
        if cfg is None:
            continue
            
        print(f"Evaluating {name}...")
        dcp = DCPDehazeFilter(cfg)
        
        ssim, psnr = evaluate_on_reside(dcp, reside_test_file)
        brisque = evaluate_on_dawn_nr(dcp, dawn_test_file)
        
        yolo_dir = f"A:/HUST_on_GitHub/ProjectCV/results/restored/{name}/dawn_fog_test"
        yolo_images_dir = os.path.join(yolo_dir, "images")
        yolo_labels_dir = os.path.join(yolo_dir, "labels")
        os.makedirs(yolo_images_dir, exist_ok=True)
        os.makedirs(yolo_labels_dir, exist_ok=True)
        
        map50 = evaluate_on_dawn_yolo(dcp, dawn_test_file, yolo_images_dir)
        
        results.append({
            "Config": name,
            "RESIDE_SSIM": ssim,
            "RESIDE_PSNR": psnr,
            "DAWN_BRISQUE": brisque,
            "DAWN_mAP50": map50
        })
        
    df = pd.DataFrame(results)
    df.to_csv("A:/HUST_on_GitHub/ProjectCV/results/final_evaluation.csv", index=False)
    print(df)

if __name__ == "__main__":
    main()
