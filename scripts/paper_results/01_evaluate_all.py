import os
import sys
import yaml
import numpy as np
import pandas as pd
import cv2
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.restoration.registry import create_filter
from src.datasets.reside_dataset import ResideDataset
from src.datasets.dawn_dataset import DawnDataset
from src.datasets.rain100H_dataset import PairedRainDataset
from src.datasets.snow100k_dataset import Snow100kDataset
from src.datasets.bsd_denoise_dataset import BSDDenoiseDataset
from src.datasets.lol_dataset import LOLDataset
from src.datasets.gopro_dataset import GoProDataset
from src.datasets.raindrop_dataset import RainDropDataset
from src.metrics.full_reference import compute_ssim, compute_psnr
from src.metrics.no_reference import compute_brisque, compute_niqe, compute_piqe, compute_entropy
from src.detection.yolo_runner import YOLOEvaluator
from src.metrics.detection import compute_mean_iou
from ultralytics import YOLO

def safe_compute(func, *args, **kwargs):
    try:
        val = func(*args, **kwargs)
        if np.isnan(val) or np.isinf(val): return None
        return val
    except Exception:
        return None

def get_basename(sample):
    if "image_rel" in sample: return sample["image_rel"]
    for key in ["hazy_path", "blur_path", "low_path", "rainy_path", "image_path"]:
        if key in sample: return os.path.basename(sample[key])
    # Fallback if everything else fails
    return str(hash(str(sample.keys()))) + ".jpg"

def cache_dataset(filter_obj, dataset, img_key, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    for sample in dataset:
        basename = get_basename(sample)
        img_path = os.path.join(cache_dir, basename)
        if not os.path.exists(img_path):
            restored = filter_obj.restore(sample[img_key]) if filter_obj else sample[img_key]
            cv2.imwrite(img_path, restored)

def eval_fr(dataset, gt_key, cache_dir):
    ssims, psnrs = [], []
    for sample in dataset:
        basename = get_basename(sample)
        img_path = os.path.join(cache_dir, basename)
        if os.path.exists(img_path):
            restored = cv2.imread(img_path)
            gt = sample[gt_key]
            s = safe_compute(compute_ssim, restored, gt)
            p = safe_compute(compute_psnr, restored, gt)
            if s is not None: ssims.append(s)
            if p is not None: psnrs.append(p)
    return np.mean(ssims) if ssims else 0, np.mean(psnrs) if psnrs else 0

def eval_nr(dataset, cache_dir):
    brisques, niqes = [], []
    for sample in dataset:
        basename = get_basename(sample)
        img_path = os.path.join(cache_dir, basename)
        if os.path.exists(img_path):
            restored = cv2.imread(img_path)
            b = safe_compute(compute_brisque, restored)
            n = safe_compute(compute_niqe, restored)
            if b is not None: brisques.append(b)
            if n is not None: niqes.append(n)
    return np.mean(brisques) if brisques else 0, np.mean(niqes) if niqes else 0

def eval_yolo(dataset, results_dir, dataset_name=""):
    images_dir = os.path.join(results_dir, "images")
    labels_dir = os.path.join(results_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    for sample in dataset:
        if sample.get("label_path") and os.path.exists(sample["label_path"]):
            dst_label = os.path.join(labels_dir, sample["label_rel"])
            if "FoggyCityscapes" in dataset_name.replace(" ", ""):
                with open(sample["label_path"], "r") as src_f, open(dst_label, "w") as dst_f:
                    for line in src_f:
                        parts = line.strip().split()
                        if not parts: continue
                        cls_id = int(parts[0])
                        # Foggy Cityscapes: 0: Car, 1: Person -> COCO: 0: Person, 2: Car
                        if cls_id == 0: cls_id = 2
                        elif cls_id == 1: cls_id = 0
                        parts[0] = str(cls_id)
                        dst_f.write(" ".join(parts) + "\n")
            else:
                shutil.copy2(sample["label_path"], dst_label)
    
    yaml_path = os.path.join(results_dir, "temp.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(results_dir)}\n")
        f.write(f"train: images\nval: images\n")
        f.write("names:\n  0: person\n  1: bicycle\n  2: car\n  3: motorcycle\n  4: airplane\n  5: bus\n  6: train\n  7: truck\n")
    
    evaluator = YOLOEvaluator("A:/HUST_on_GitHub/ProjectCV/yolo26n.pt", yaml_path)
    
    unique_classes = set()
    for f in os.listdir(labels_dir):
        if f.endswith('.txt'):
            with open(os.path.join(labels_dir, f), 'r') as txt:
                for line in txt:
                    parts = line.strip().split()
                    if parts:
                        unique_classes.add(int(parts[0]))
    unique_classes = sorted(list(unique_classes))
    if not unique_classes:
        unique_classes = [0, 1, 2, 3, 5, 7]
        
    metrics = evaluator.evaluate(yaml_path, classes=unique_classes) or {}
    
    yolo_model = YOLO("A:/HUST_on_GitHub/ProjectCV/yolo26n.pt")
    mean_iou = compute_mean_iou(yolo_model, images_dir, labels_dir, allowed_classes=unique_classes)
    metrics['mean_iou'] = mean_iou
    return metrics

def run_evaluation():
    base_dir = "A:/HUST_on_GitHub/ProjectCV"
    paper_tables_dir = os.path.join(base_dir, "paper", "tables")
    paper_figures_dir = os.path.join(base_dir, "paper", "figures", "detection")
    os.makedirs(paper_tables_dir, exist_ok=True)
    os.makedirs(paper_figures_dir, exist_ok=True)

    configs = {
        "DCP_DAWN": {
            "type": "dcp", "img_fr": "hazy", "gt_fr": "clear",
            "configs": {
                "Raw": None,
                "Default": "configs/dcp/dcp_default.yaml",
                "Config1": "configs/dcp/optuna_config1_reside_ssim.yaml",
                "Config2": "configs/dcp/optuna_config2_dawn_brisque.yaml",
                "Config3": "configs/dcp/optuna_config3_dawn_map50.yaml"
            },
            "ds_fr": ResideDataset(os.path.join(base_dir, "data/reside6k/splits/test.txt"), os.path.join(base_dir, "dataset/RESIDE-6K")),
            "ds_nr": DawnDataset(os.path.join(base_dir, "data/dawn_fog/splits/fog_test_pairs.csv"), "")
        },
        "DCP_RTTS": {
            "type": "dcp", "img_fr": None, "gt_fr": None,
            "configs": {
                "Raw": None,
                "Default": "configs/dcp/dcp_default.yaml",
                "Config1": "configs/dcp/optuna_config1_reside_ssim.yaml",
                "Config2": "configs/dcp/optuna_config2_dawn_brisque.yaml",
                "Config3": "configs/dcp/optuna_config3_dawn_map50.yaml"
            },
            "ds_fr": None,
            "ds_nr": DawnDataset(os.path.join(base_dir, "data/rtts/splits/val_test_pairs.csv"), "")
        },
        "DCP_FoggyCityscapes": {
            "type": "dcp", "img_fr": None, "gt_fr": None,
            "configs": {
                "Raw": None,
                "Default": "configs/dcp/dcp_default.yaml",
                "Config1": "configs/dcp/optuna_config1_reside_ssim.yaml",
                "Config2": "configs/dcp/optuna_config2_dawn_brisque.yaml",
                "Config3": "configs/dcp/optuna_config3_dawn_map50.yaml"
            },
            "ds_fr": None,
            "ds_nr": DawnDataset(os.path.join(base_dir, "data/foggycityscapes/splits/val_test_pairs.csv"), "")
        },
        "WMGF": {
            "type": "wmgf", "img_fr": "rainy", "gt_fr": "clean",
            "configs": {
                "Raw": None,
                "Default": "configs/wmgf/derain_wmgf_default.yaml",
                "Config1": "configs/wmgf/derain_optuna_config1_ssim.yaml",
                "Config2": "configs/wmgf/derain_optuna_config2_brisque.yaml",
                "Config3": "configs/wmgf/derain_optuna_config3_map50.yaml"
            },
            "ds_fr": PairedRainDataset(os.path.join(base_dir, "data/rain100h/splits/test.txt"), os.path.join(base_dir, "dataset/rain100H")),
            "ds_nr": DawnDataset(os.path.join(base_dir, "data/dawn_rain/splits/rain_test_pairs.csv"), "")
        },
        "WMGF_RainDrop": {
            "type": "wmgf", "img_fr": "rainy", "gt_fr": "clean",
            "configs": {
                "Raw": None,
                "Default": "configs/wmgf/derain_wmgf_default.yaml",
                "Config1": "configs/wmgf/derain_optuna_config1_ssim.yaml",
                "Config2": "configs/wmgf/derain_optuna_config2_brisque.yaml",
                "Config3": "configs/wmgf/derain_optuna_config3_map50.yaml"
            },
            "ds_fr": RainDropDataset(os.path.join(base_dir, "dataset/RainDrop")),
            "ds_nr": None
        },
        "Desnow": {
            "type": "desnow", "img_fr": "synthetic", "gt_fr": "clear",
            "configs": {
                "Raw": None,
                "Default": "configs/desnowing/desnow_morph_guided_default.yaml",
                "Config1": "configs/desnowing/optuna_config1_ssim.yaml",
                "Config2": "configs/desnowing/optuna_config2_brisque.yaml",
                "Config3": "configs/desnowing/optuna_config3_map50.yaml"
            },
            "ds_fr": Snow100kDataset(os.path.join(base_dir, "data/snow100k/splits/test.txt"), os.path.join(base_dir, "dataset/Snow100K")),
            "ds_nr": DawnDataset(os.path.join(base_dir, "data/dawn/splits/snow_test_pairs.csv"), "")
        },
        "LIME": {
            "type": "lime", "img_fr": "low", "gt_fr": "normal",
            "configs": {
                "Raw": None,
                "Default": "configs/lime/lime_delowlight_default.yaml",
                "Config1": "configs/lime/lime_delowlight_config1_ssim.yaml",
                "Config2": "configs/lime/lime_delowlight_config2_brisque.yaml",
                "Config3": None
            },
            "ds_fr": LOLDataset(os.path.join(base_dir, "data/lol/splits/test.txt"), os.path.join(base_dir, "dataset/LOL")),
            "ds_nr": None
        },
        "BM3D": {
            "type": "bm3d", "img_fr": "noisy", "gt_fr": "clean",
            "configs": {
                "Raw": None,
                "Default": "configs/bm3d/bm3d_default_noise25.yaml",
                "Config1": "configs/bm3d/bm3d_config1_bsd25_ssim.yaml",
                "Config2": "configs/bm3d/bm3d_config2_bsd25_brisque.yaml",
                "Config3": None
            },
            "ds_fr": BSDDenoiseDataset(root=os.path.join(base_dir, "data/bsd_denoise"), noise_level=25, split='test'),
            "ds_nr": None
        },
        "RBCP": {
            "type": "rbcp", "img_fr": None, "gt_fr": None,
            "configs": {
                "Raw": None,
                "Default": "configs/rbcp/desand_rbcp_default.yaml",
                "Config1": None,
                "Config2": "configs/rbcp/desand_optuna_config2_brisque.yaml",
                "Config3": "configs/rbcp/desand_optuna_config3_map50.yaml"
            },
            "ds_fr": None,
            "ds_nr": DawnDataset(os.path.join(base_dir, "data/dawn/splits/sand_test_pairs.csv"), "")
        },
        "MotionDeblur": {
            "type": "motion_deblur", "img_fr": "blur", "gt_fr": "sharp",
            "configs": {
                "Raw": None,
                "Default": "configs/motion_deblur/richardson_lucy_default.yaml",
                "Config1": "configs/motion_deblur/optuna_config1_ssim.yaml",
                "Config2": "configs/motion_deblur/optuna_config2_brisque.yaml",
                "Config3": None
            },
            "ds_fr": GoProDataset(os.path.join(base_dir, "outputs/motion_deblur/gopro_pairs.csv"), split='test'),
            "ds_nr": None
        }
    }

    all_results = []
    
    for f_name, f_data in configs.items():
        for c_name, c_path in f_data["configs"].items():
            if c_path is None and c_name != "Raw": continue
            
            print(f"Evaluating {f_name} - {c_name}...")
            if c_name == "Raw":
                filt = None
            else:
                full_path = os.path.join(base_dir, c_path)
                if not os.path.exists(full_path):
                    print(f"  Missing config: {full_path}")
                    continue
                filt = create_filter(f_data["type"], full_path)

            res = {"Filter": f_name, "Config": c_name}
            
            # Phase 1
            if f_data["ds_fr"]:
                print("  Phase 1: FR Metrics")
                fr_cache_dir = os.path.join(paper_figures_dir, f_name, c_name, "fr_images")
                cache_dataset(filt, f_data["ds_fr"], f_data["img_fr"], fr_cache_dir)
                ssim, psnr = eval_fr(f_data["ds_fr"], f_data["gt_fr"], fr_cache_dir)
                res["Phase1_SSIM"] = ssim
                res["Phase1_PSNR"] = psnr
                
            # Phase 2
            if f_data["ds_nr"]:
                print("  Phase 2: NR Metrics")
                nr_cache_dir = os.path.join(paper_figures_dir, f_name, c_name, "images")
                cache_dataset(filt, f_data["ds_nr"], "image", nr_cache_dir)
                brisque, niqe = eval_nr(f_data["ds_nr"], nr_cache_dir)
                res["Phase2_BRISQUE"] = brisque
                res["Phase2_NIQE"] = niqe
                
                print("  Phase 2: Detection")
                yolo_dir = os.path.join(paper_figures_dir, f_name, c_name)
                y_met = eval_yolo(f_data["ds_nr"], yolo_dir, dataset_name=f_name)
                res["Phase2_mAP50"] = y_met.get("map50", 0)
                res["Phase2_mAP50-95"] = y_met.get("map50_95", 0)
                res["Phase2_meanIoU"] = y_met.get("mean_iou", 0)
                
            all_results.append(res)
            
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(paper_tables_dir, "all_metrics_summary.csv"), index=False)
    
    # Save to LaTeX
    with open(os.path.join(paper_tables_dir, "metrics_table.tex"), "w") as f:
        f.write(df.to_latex(index=False, float_format="%.4f"))
        
    print("ALL EVALUATIONS DONE.")

if __name__ == "__main__":
    run_evaluation()
