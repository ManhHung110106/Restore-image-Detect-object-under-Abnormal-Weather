import os
import sys
import numpy as np
import yaml
import json
import uuid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.datasets.snow100k_dataset import Snow100kDataset
from src.datasets.dawn_dataset import DawnDataset
from src.metrics.full_reference import compute_ssim, compute_psnr
from src.metrics.no_reference import compute_brisque, compute_niqe
from src.detection.yolo_runner import YOLOEvaluator
from src.restoration.registry import create_filter

def eval_full_reference(filter_obj, dataset):
    ssims = []
    psnrs = []
    for sample in dataset:
        restored = filter_obj.restore(sample["synthetic"]) if filter_obj else sample["synthetic"]
        clear = sample["clear"]
        ssims.append(compute_ssim(restored, clear))
        psnrs.append(compute_psnr(restored, clear))
    return np.mean(ssims), np.mean(psnrs)

def eval_no_reference(filter_obj, dataset):
    brisques = []
    niqes = []
    for sample in dataset:
        restored = filter_obj.restore(sample["image"]) if filter_obj else sample["image"]
        b = compute_brisque(restored)
        n = compute_niqe(restored)
        if not np.isnan(b): brisques.append(b)
        if not np.isnan(n): niqes.append(n)
    return np.mean(brisques), np.mean(niqes)

def eval_detection(filter_obj, dataset, yolo_evaluator):
    import cv2
    import shutil
    
    hash_id = str(uuid.uuid4().hex)[:8]
    tmp_dir = os.path.join("A:/HUST_on_GitHub/ProjectCV/results/tmp_eval", hash_id)
    images_dir = os.path.join(tmp_dir, "images")
    labels_dir = os.path.join(tmp_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    val_txt = os.path.join(tmp_dir, "val.txt")
    with open(val_txt, "w") as f_val:
        for sample in dataset:
            img_path = sample["image_path"]
            basename = os.path.basename(img_path)
            
            if filter_obj:
                restored = filter_obj.restore(sample["image"])
                out_path = os.path.join(images_dir, basename)
                cv2.imwrite(out_path, restored)
            else:
                out_path = os.path.join(images_dir, basename)
                shutil.copy2(img_path, out_path)
                
            f_val.write(f"{out_path}\n")
            
            lbl_name = os.path.splitext(basename)[0] + ".txt"
            lbl_path = sample.get("label_path", None)
            out_lbl = os.path.join(labels_dir, lbl_name)
            if lbl_path and os.path.exists(lbl_path):
                shutil.copy2(lbl_path, out_lbl)
            else:
                open(out_lbl, 'w').close()
                
    yaml_path = os.path.join(tmp_dir, "dataset.yaml")
    with open(yaml_path, "w") as f_yaml:
        f_yaml.write(f"path: {tmp_dir}\n")
        f_yaml.write(f"train: val.txt\n")
        f_yaml.write(f"val: val.txt\n")
        f_yaml.write("names:\n")
        classes = ["bicycle", "bus", "car", "motorcycle", "person", "train", "truck"]
        for i, cls in enumerate(classes):
            f_yaml.write(f"  {i}: {cls}\n")
            
    yolo_evaluator.data_yaml = yaml_path
    results = yolo_evaluator.evaluate(yaml_path)
    
    if not results:
        return 0.0, 0.0
        
    return results.get("map50", 0.0), results.get("map50_95", 0.0)
    
def main():
    snow_test = Snow100kDataset("A:/HUST_on_GitHub/ProjectCV/data/snow100k/splits/test.txt", "A:/HUST_on_GitHub/ProjectCV/dataset/Snow100K")
    dawn_test = DawnDataset("A:/HUST_on_GitHub/ProjectCV/data/dawn/splits/snow_test_pairs.csv", "")
    yolo_eval = YOLOEvaluator(model_path="A:/HUST_on_GitHub/ProjectCV/yolo26n.pt", data_yaml="", device="auto")
    
    configs = {
        "raw": None,
        "default": "A:/HUST_on_GitHub/ProjectCV/configs/desnow_morph_guided_default.yaml",
        "config1": "A:/HUST_on_GitHub/ProjectCV/results/optuna/desnow_config1_ssim/best_config.yaml",
        "config2": "A:/HUST_on_GitHub/ProjectCV/results/optuna/desnow_config2_brisque/best_config.yaml",
        "config3": "A:/HUST_on_GitHub/ProjectCV/results/optuna/desnow_config3_map50/best_config.yaml"
    }
    
    results = {}
    for name, path in configs.items():
        print(f"--- Evaluating {name} ---")
        if path is None:
            filter_obj = None
        else:
            if not os.path.exists(path):
                print(f"Skipping {name}, config not found.")
                continue
            filter_obj = create_filter("desnow", path)
            
        ssim, psnr = eval_full_reference(filter_obj, snow_test)
        brisque, niqe = eval_no_reference(filter_obj, dawn_test)
        map50, map50_95 = eval_detection(filter_obj, dawn_test, yolo_eval)
        
        results[name] = {
            "ssim": float(ssim),
            "psnr": float(psnr),
            "brisque": float(brisque),
            "niqe": float(niqe),
            "map50": float(map50),
            "map50_95": float(map50_95)
        }
        
    os.makedirs("A:/HUST_on_GitHub/ProjectCV/results/desnow_evaluation", exist_ok=True)
    with open("A:/HUST_on_GitHub/ProjectCV/results/desnow_evaluation/metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Evaluation complete. Results saved to A:/HUST_on_GitHub/ProjectCV/results/desnow_evaluation/metrics.json")
    
if __name__ == "__main__":
    main()
