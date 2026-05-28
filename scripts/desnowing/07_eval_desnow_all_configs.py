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
from src.metrics.no_reference import compute_brisque, compute_niqe, compute_piqe, compute_entropy
from src.detection.yolo_runner import YOLOEvaluator
import pandas as pd
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
    piqes = []
    entropies = []
    for sample in dataset:
        restored = filter_obj.restore(sample["image"]) if filter_obj else sample["image"]
        b = compute_brisque(restored)
        n = compute_niqe(restored)
        p = compute_piqe(restored)
        e = compute_entropy(restored)
        if not np.isnan(b): brisques.append(b)
        if not np.isnan(n): niqes.append(n)
        if not np.isnan(p): piqes.append(p)
        if not np.isnan(e): entropies.append(e)
    return np.mean(brisques), np.mean(niqes), np.mean(piqes), np.mean(entropies)

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
        coco_names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train", 7: "truck"}
        for k, v in coco_names.items():
            f_yaml.write(f"  {k}: {v}\n")
            
    yolo_evaluator.data_yaml = yaml_path
    results = yolo_evaluator.evaluate(yaml_path)
    
    from src.metrics.detection import compute_mean_iou
    mean_iou = compute_mean_iou(yolo_evaluator.model, images_dir, labels_dir, allowed_classes=[0, 1, 2, 3, 5, 6, 7])
    if results:
        results["mean_iou"] = mean_iou
    
    if not results:
        return 0.0, 0.0, 0.0, 0.0, 0.0
        
    return (
        results.get("map50", 0.0), 
        results.get("map50_95", 0.0),
        results.get("mean_iou", 0.0),
        results.get("precision", 0.0),
        results.get("recall", 0.0)
    )
    
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
    
    results_list = []
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
        brisque, niqe, piqe, entropy = eval_no_reference(filter_obj, dawn_test)
        map50, map50_95, mean_iou, precision, recall = eval_detection(filter_obj, dawn_test, yolo_eval)
        
        results_list.append({
            "Config": name,
            "Snow100K_SSIM": float(ssim),
            "Snow100K_PSNR": float(psnr),
            "DAWN_BRISQUE": float(brisque),
            "DAWN_NIQE": float(niqe),
            "DAWN_PIQE": float(piqe),
            "DAWN_Entropy": float(entropy),
            "DAWN_mAP50": float(map50),
            "DAWN_mAP50-95": float(map50_95),
            "DAWN_IoU": float(mean_iou),
            "DAWN_Precision": float(precision),
            "DAWN_Recall": float(recall)
        })
        
    os.makedirs("A:/HUST_on_GitHub/ProjectCV/results/desnow_evaluation", exist_ok=True)
    df = pd.DataFrame(results_list)
    csv_path = "A:/HUST_on_GitHub/ProjectCV/results/desnow_evaluation/metrics.csv"
    df.to_csv(csv_path, index=False)
    
    print("\n--- FINAL EVALUATION RESULTS ---")
    print(df.to_string())
    print(f"Evaluation complete. Results saved to {csv_path}")
    
if __name__ == "__main__":
    main()
