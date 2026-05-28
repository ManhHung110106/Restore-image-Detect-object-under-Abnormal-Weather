import os
import cv2
import json
import csv
import sys
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import shutil

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)
os.chdir(project_root)

from src.restoration.registry import create_filter
from src.metrics.no_reference import compute_brisque, compute_niqe, compute_piqe, compute_entropy
from src.metrics.detection import extract_detection_metrics, compute_mean_iou

def load_pairs(csv_path):
    pairs = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 2:
                pairs.append((row[0], row[1]))
    return pairs

def main():
    test_csv = "data/dawn/splits/sand_test_pairs.csv"
    dataset_yaml = "data/dawn/dawn_sand.yaml"
    
    if not os.path.exists(test_csv):
        print("Please run 01_prepare_dawn_sand.py first")
        return
        
    pairs = load_pairs(test_csv)
    yolo_model = YOLO("yolov8n.pt") # YOLO26 stands for a generic standard model, falling back to yolov8n.pt if v26 isn't a thing
    
    # We evaluate 4 configs:
    # 1. Raw
    # 2. RBCP Default
    # 3. RBCP Config2 (BRISQUE)
    # 4. RBCP Config3 (mAP50)
    
    configs = [
        ("Raw", None),
        ("RBCP_Default", "configs/desand_rbcp_default.yaml")
    ]
    
    if os.path.exists("configs/desand_optuna_config2_brisque.yaml"):
        configs.append(("Desand-config2-BRISQUE", "configs/desand_optuna_config2_brisque.yaml"))
    
    if os.path.exists("configs/desand_optuna_config3_map50.yaml"):
        configs.append(("Desand-config3-mAP50", "configs/desand_optuna_config3_map50.yaml"))

    all_metrics = {}
    
    for name, cfg_path in configs:
        print(f"\nEvaluating {name}...")
        
        # Prepare restored images
        cache_dir = f"data/dawn/cache/eval_{name}"
        cache_img_dir = os.path.join(cache_dir, "images")
        cache_lbl_dir = os.path.join(cache_dir, "labels")
        cache_yaml_path = os.path.join(cache_dir, "dataset.yaml")
        
        os.makedirs(cache_img_dir, exist_ok=True)
        os.makedirs(cache_lbl_dir, exist_ok=True)
        
        restorer = None
        if cfg_path is not None:
            restorer = create_filter("desand", cfg_path)
            
        nr_metrics = defaultdict(list)
        
        for img_path, lbl_path in pairs:
            img = cv2.imread(img_path)
            if img is None: continue
                
            if restorer is not None:
                restored = restorer.restore(img)
            else:
                restored = img
                
            # Compute no-reference metrics
            nr_metrics["brisque"].append(compute_brisque(restored))
            nr_metrics["niqe"].append(compute_niqe(restored))
            nr_metrics["piqe"].append(compute_piqe(restored))
            nr_metrics["entropy"].append(compute_entropy(restored))
            
            # Save for YOLO
            basename = os.path.basename(img_path)
            out_img = os.path.join(cache_img_dir, basename)
            cv2.imwrite(out_img, restored)
            
            lbl_basename = os.path.basename(lbl_path)
            out_lbl = os.path.join(cache_lbl_dir, lbl_basename)
            if os.path.exists(lbl_path):
                shutil.copy(lbl_path, out_lbl)
                
        # Generate yaml for yolo evaluation
        with open(dataset_yaml, 'r') as f:
            lines = f.readlines()
            
        with open(cache_yaml_path, 'w') as f:
            for line in lines:
                if line.startswith("path:"):
                    f.write(f"path: {os.path.abspath(cache_dir)}\n")
                elif line.startswith("test:"):
                    f.write("test: images\n") 
                elif line.startswith("val:"):
                    f.write("val: images\n")
                elif line.startswith("train:"):
                    f.write("train: images\n")
                else:
                    f.write(line)
                    
        # YOLO evaluation
        results = yolo_model.val(data=cache_yaml_path, split="test", save_json=False, save=False, plots=False, verbose=False)
        det_metrics = extract_detection_metrics(results)
        
        mean_iou = compute_mean_iou(yolo_model, cache_img_dir, cache_lbl_dir, allowed_classes=[0, 1, 2, 3, 5, 6, 7])
        
        # Aggregate metrics
        agg_nr = {k: np.nanmean(v) for k, v in nr_metrics.items()}
        
        all_metrics[name] = {
            "no_reference": agg_nr,
            "detection": {**det_metrics, "mean_iou": mean_iou}
        }
        
        shutil.rmtree(cache_dir)
        
    # Save metrics
    os.makedirs("results/metrics", exist_ok=True)
    
    with open("results/metrics/desand_all_metrics_summary.json", "w") as f:
        json.dump(all_metrics, f, indent=4)
        
    with open("results/metrics/desand_no_reference_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "BRISQUE", "NIQE", "PIQE", "Entropy"])
        for name, m in all_metrics.items():
            nr = m["no_reference"]
            writer.writerow([name, nr.get("brisque", ""), nr.get("niqe", ""), nr.get("piqe", ""), nr.get("entropy", "")])
            
    with open("results/metrics/desand_detection_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        # We don't know exact classes ap ahead of time, let's just get keys from the first one
        det_keys = ["map50", "map50_95", "mean_iou", "precision", "recall", "f1"]
        for k in all_metrics[configs[0][0]]["detection"].keys():
            if k.startswith("ap_"): det_keys.append(k)
            
        writer.writerow(["Method"] + det_keys)
        for name, m in all_metrics.items():
            det = m["detection"]
            writer.writerow([name] + [det.get(k, "") for k in det_keys])
            
    print("Evaluation complete. Metrics saved to results/metrics/")

if __name__ == "__main__":
    main()
