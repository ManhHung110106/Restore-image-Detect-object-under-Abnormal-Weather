import os
import sys
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

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
from ultralytics import YOLO

base_dir = "A:/HUST_on_GitHub/ProjectCV"
paper_figs = os.path.join(base_dir, "paper", "figures")
os.makedirs(paper_figs, exist_ok=True)
yolo_model = YOLO("A:/HUST_on_GitHub/ProjectCV/yolo26n.pt")

configs_def = {
    "DCP_DAWN": {
        "type": "dcp", "p1_img": "hazy", "p1_gt": "clear", "p2_img": "image",
        "p1_ds_name": "RESIDE", "p2_ds_name": "DAWN Fog",
        "cfgs": {
            "Default": "configs/dcp/dcp_default.yaml",
            "Config 1": "configs/dcp/optuna_config1_reside_ssim.yaml",
            "Config 2": "configs/dcp/optuna_config2_dawn_brisque.yaml",
            "Config 3": "configs/dcp/optuna_config3_dawn_map50.yaml"
        },
        "ds1": ResideDataset(os.path.join(base_dir, "data/reside6k/splits/test.txt"), os.path.join(base_dir, "dataset/RESIDE-6K")),
        "ds2": DawnDataset(os.path.join(base_dir, "data/dawn_fog/splits/fog_test_pairs.csv"), "")
    },
    "DCP_RTTS": {
        "type": "dcp", "p1_img": "hazy", "p1_gt": "clear", "p2_img": "image",
        "p1_ds_name": "RESIDE", "p2_ds_name": "RTTS",
        "cfgs": {
            "Default": "configs/dcp/dcp_default.yaml",
            "Config 1": "configs/dcp/optuna_config1_reside_ssim.yaml",
            "Config 2": "configs/dcp/optuna_config2_dawn_brisque.yaml",
            "Config 3": "configs/dcp/optuna_config3_dawn_map50.yaml"
        },
        "ds1": None,
        "ds2": DawnDataset(os.path.join(base_dir, "data/rtts/splits/val_test_pairs.csv"), "")
    },
    "DCP_FoggyCityscapes": {
        "type": "dcp", "p1_img": "hazy", "p1_gt": "clear", "p2_img": "image",
        "p1_ds_name": "RESIDE", "p2_ds_name": "Foggy Cityscapes",
        "cfgs": {
            "Default": "configs/dcp/dcp_default.yaml",
            "Config 1": "configs/dcp/optuna_config1_reside_ssim.yaml",
            "Config 2": "configs/dcp/optuna_config2_dawn_brisque.yaml",
            "Config 3": "configs/dcp/optuna_config3_dawn_map50.yaml"
        },
        "ds1": None,
        "ds2": DawnDataset(os.path.join(base_dir, "data/foggycityscapes/splits/val_test_pairs.csv"), "")
    },
    "WMGF": {
        "type": "wmgf", "p1_img": "rainy", "p1_gt": "clean", "p2_img": "image",
        "p1_ds_name": "Rain100H", "p2_ds_name": "DAWN Rain",
        "cfgs": {
            "Default": "configs/wmgf/derain_wmgf_default.yaml",
            "Config 1": "configs/wmgf/derain_optuna_config1_ssim.yaml",
            "Config 2": "configs/wmgf/derain_optuna_config2_brisque.yaml",
            "Config 3": "configs/wmgf/derain_optuna_config3_map50.yaml"
        },
        "ds1": PairedRainDataset(os.path.join(base_dir, "data/rain100h/splits/test.txt"), os.path.join(base_dir, "dataset/rain100H")),
        "ds2": DawnDataset(os.path.join(base_dir, "data/dawn_rain/splits/rain_test_pairs.csv"), "")
    },
    "WMGF_RainDrop": {
        "type": "wmgf", "p1_img": "rainy", "p1_gt": "clean", "p2_img": None,
        "p1_ds_name": "RainDrop", "p2_ds_name": "",
        "cfgs": {
            "Default": "configs/wmgf/derain_wmgf_default.yaml",
            "Config 1": "configs/wmgf/derain_optuna_config1_ssim.yaml",
            "Config 2": "configs/wmgf/derain_optuna_config2_brisque.yaml",
            "Config 3": "configs/wmgf/derain_optuna_config3_map50.yaml"
        },
        "ds1": RainDropDataset(os.path.join(base_dir, "dataset/RainDrop")),
        "ds2": None
    },
    "Desnow": {
        "type": "desnow", "p1_img": "synthetic", "p1_gt": "clear", "p2_img": "image",
        "p1_ds_name": "Snow100K", "p2_ds_name": "DAWN Snow",
        "cfgs": {
            "Default": "configs/desnowing/desnow_morph_guided_default.yaml",
            "Config 1": "configs/desnowing/optuna_config1_ssim.yaml",
            "Config 2": "configs/desnowing/optuna_config2_brisque.yaml",
            "Config 3": "configs/desnowing/optuna_config3_map50.yaml"
        },
        "ds1": Snow100kDataset(os.path.join(base_dir, "data/snow100k/splits/test.txt"), os.path.join(base_dir, "dataset/Snow100K")),
        "ds2": DawnDataset(os.path.join(base_dir, "data/dawn/splits/snow_test_pairs.csv"), "")
    },
    "LIME": {
        "type": "lime", "p1_img": "low", "p1_gt": "normal", "p2_img": None,
        "p1_ds_name": "LOL", "p2_ds_name": "",
        "cfgs": {
            "Default": "configs/lime/lime_delowlight_default.yaml",
            "Config 1": "configs/lime/lime_delowlight_config1_ssim.yaml",
            "Config 2": "configs/lime/lime_delowlight_config2_brisque.yaml",
            "Config 3": None
        },
        "ds1": LOLDataset(os.path.join(base_dir, "data/lol/splits/test.txt"), os.path.join(base_dir, "dataset/LOL")),
        "ds2": None
    },
    "BM3D": {
        "type": "bm3d", "p1_img": "noisy", "p1_gt": "clean", "p2_img": None,
        "p1_ds_name": "BSD25", "p2_ds_name": "",
        "cfgs": {
            "Default": "configs/bm3d/bm3d_default_noise25.yaml",
            "Config 1": "configs/bm3d/bm3d_config1_bsd25_ssim.yaml",
            "Config 2": "configs/bm3d/bm3d_config2_bsd25_brisque.yaml",
            "Config 3": None
        },
        "ds1": BSDDenoiseDataset(root=os.path.join(base_dir, "data/bsd_denoise"), noise_level=25, split='test'),
        "ds2": None
    },
    "RBCP": {
        "type": "rbcp", "p1_img": None, "p1_gt": None, "p2_img": "image",
        "p1_ds_name": "", "p2_ds_name": "DAWN Sand",
        "cfgs": {
            "Default": "configs/rbcp/desand_rbcp_default.yaml",
            "Config 1": None,
            "Config 2": "configs/rbcp/desand_optuna_config2_brisque.yaml",
            "Config 3": "configs/rbcp/desand_optuna_config3_map50.yaml"
        },
        "ds1": None,
        "ds2": DawnDataset(os.path.join(base_dir, "data/dawn/splits/sand_test_pairs.csv"), "")
    },
    "MotionDeblur": {
        "type": "motion_deblur", "p1_img": "blur", "p1_gt": "sharp", "p2_img": None,
        "p1_ds_name": "GoPro", "p2_ds_name": "",
        "cfgs": {
            "Default": "configs/motion_deblur/richardson_lucy_default.yaml",
            "Config 1": "configs/motion_deblur/optuna_config1_ssim.yaml",
            "Config 2": "configs/motion_deblur/optuna_config2_brisque.yaml",
            "Config 3": None
        },
        "ds1": GoProDataset(os.path.join(base_dir, "outputs/motion_deblur/gopro_pairs.csv"), split='test'),
        "ds2": None
    }
}

def load_filt(f_type, path):
    if not path or not os.path.exists(os.path.join(base_dir, path)): return None
    return create_filter(f_type, os.path.join(base_dir, path))

def run_yolo_draw(img):
    res = yolo_model.predict(img, verbose=False, classes=[0, 1, 2, 3, 5, 7], conf=0.1)[0]
    num_bb = len(res.boxes)
    ann_img = res.plot()
    return ann_img, num_bb

CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

def draw_gt_bboxes(img, label_path, dataset_name=""):
    if not label_path or not os.path.exists(label_path): return img, 0
    h, w = img.shape[:2]
    img_cp = img.copy()
    count = 0
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                count += 1
                cls_id = int(parts[0])
                
                # Remap dataset native IDs to COCO
                if "FoggyCityscapes" in dataset_name.replace(" ", ""):
                    if cls_id == 0: cls_id = 2  # Car
                    elif cls_id == 1: cls_id = 0 # Person
                    
                cls_name = CLASS_NAMES.get(cls_id, str(cls_id))
                
                if len(parts) == 5:
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                else:
                    coords = list(map(float, parts[1:]))
                    xs = coords[0::2]
                    ys = coords[1::2]
                    x1 = int(min(xs) * w)
                    x2 = int(max(xs) * w)
                    y1 = int(min(ys) * h)
                    y2 = int(max(ys) * h)
                cv2.rectangle(img_cp, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(img_cp, cls_name, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img_cp, count

def create_grid(phase, ds_key, rows, cols_filters, filename, detection=False):
    fig, axes = plt.subplots(len(rows), len(cols_filters), figsize=(3 * len(cols_filters), 3 * len(rows)))
    if len(rows) == 1: axes = [axes]
    
    for c_idx, f_name in enumerate(cols_filters):
        f_data = configs_def[f_name]
        ds = f_data[ds_key]
        if ds is None or len(ds) == 0:
            for r in range(len(rows)): axes[r][c_idx].axis('off')
            continue
            
        sample_idx = random.randint(0, len(ds) - 1)
        sample = ds[sample_idx]
        img_key = f_data["p1_img"] if phase==1 else f_data["p2_img"]
        gt_key = f_data.get("p1_gt") if phase==1 else None
        
        raw_img = sample[img_key]
        clean_img = sample[gt_key] if gt_key else None
        
        for r_idx, row_name in enumerate(rows):
            ax = axes[r_idx][c_idx] if len(rows) > 1 else axes[c_idx]
            ax.axis('off')
            
            # Title for first row
            if r_idx == 0:
                ax.set_title(f"{f_name}\n({f_data['p1_ds_name'] if phase==1 else f_data['p2_ds_name']})", fontsize=14, fontweight='bold')
                
            # Row labels
            if c_idx == 0:
                ax.text(-0.1, 0.5, row_name, va='center', ha='right', rotation=90, fontsize=14, fontweight='bold', transform=ax.transAxes)
                
            img_to_show = None
            text_to_show = ""
            
            if row_name == "Clean (Thực)":
                img_to_show = clean_img
            elif row_name == "Raw (Hỏng)":
                img_to_show = raw_img
            elif row_name == "GT Bounding Box":
                img_to_show = raw_img
                if detection:
                    img_to_show, num_bb = draw_gt_bboxes(raw_img, sample.get("label_path"), f_data.get("p2_ds_name", ""))
                    text_to_show = f"BBs: {num_bb}"
            elif row_name == "Raw (YOLO)":
                img_to_show = raw_img
                if detection:
                    img_to_show, num_bb = run_yolo_draw(img_to_show)
                    text_to_show = f"BBs: {num_bb}"
            else: # Các Config
                cfg_path = f_data["cfgs"].get(row_name)
                filt = load_filt(f_data["type"], cfg_path)
                if filt:
                    img_to_show = filt.restore(raw_img)
                    if detection:
                        img_to_show, num_bb = run_yolo_draw(img_to_show)
                        text_to_show = f"BBs: {num_bb}"
                        
            if img_to_show is not None:
                # RGB convert for matplotlib
                if len(img_to_show.shape) == 2: img_to_show = cv2.cvtColor(img_to_show, cv2.COLOR_GRAY2RGB)
                elif img_to_show.shape[2] == 3: img_to_show = cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)
                ax.imshow(img_to_show)
                if text_to_show:
                    ax.text(0.5, -0.1, text_to_show, va='top', ha='center', fontsize=12, transform=ax.transAxes, color='black', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            else:
                ax.text(0.5, 0.5, "N/A", va='center', ha='center', fontsize=12, transform=ax.transAxes)
            
    plt.tight_layout()
    plt.savefig(os.path.join(paper_figs, filename), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    rows_p1 = ["Clean (Thực)", "Raw (Hỏng)", "Default", "Config 1", "Config 2", "Config 3"]
    rows_p2_res = ["Raw (Hỏng)", "Default", "Config 1", "Config 2", "Config 3"]
    rows_p2_det = ["GT Bounding Box", "Raw (YOLO)", "Default", "Config 1", "Config 2", "Config 3"]
    
    for v in range(1, 6):
        print(f"Generating Version {v}...")
        random.seed(42 + v)
        create_grid(1, "ds1", rows_p1, ["DCP_DAWN", "WMGF", "WMGF_RainDrop", "Desnow", "LIME", "BM3D", "MotionDeblur"], f"phase1_restore_grid_v{v}.png")
        create_grid(2, "ds2", rows_p2_res, ["DCP_DAWN", "DCP_RTTS", "DCP_FoggyCityscapes", "WMGF", "Desnow", "RBCP"], f"phase2_restore_grid_v{v}.png")
        create_grid(2, "ds2", rows_p2_det, ["DCP_DAWN", "DCP_RTTS", "DCP_FoggyCityscapes", "WMGF", "Desnow", "RBCP"], f"phase2_detection_grid_v{v}.png", detection=True)
    
    print("All visualizations saved successfully in paper/figures/")

if __name__ == "__main__":
    main()
