import os
import yaml
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.datasets.gopro_dataset import GoProDataset
from src.restoration.wiener_deblur import WienerMotionDeblurFilter
from src.restoration.richardson_lucy_deblur import RichardsonLucyMotionDeblurFilter
from src.restoration.fergus_blind_deblur import FergusBlindMotionDeblurFilter

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

import json

def get_filter_from_optuna(json_path):
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r') as f:
        params = json.load(f)
        
    method = params.pop('method')
    if method == 'fergus_blind':
        params['per_image_search'] = False
        
    params['clip_output'] = True
    if method == "wiener":
        params['post_sharpen'] = True if params.get('post_sharpen_amount', 0) > 0 else False
        return WienerMotionDeblurFilter(params)
    elif method == "richardson_lucy":
        return RichardsonLucyMotionDeblurFilter(params)
    elif method == "fergus_blind":
        return FergusBlindMotionDeblurFilter(params)
    return None

def main():
    base_dir = r"A:\HUST_on_GitHub\ProjectCV"
    
    wiener_cfg = load_config(os.path.join(base_dir, "configs", "motion_deblur", "wiener_default.yaml"))
    rl_cfg = load_config(os.path.join(base_dir, "configs", "motion_deblur", "richardson_lucy_default.yaml"))
    fergus_cfg = load_config(os.path.join(base_dir, "configs", "motion_deblur", "fergus_blind_default.yaml"))
    
    filters = {
        "Wiener_Default": WienerMotionDeblurFilter(wiener_cfg),
        "RichardsonLucy_Default": RichardsonLucyMotionDeblurFilter(rl_cfg),
        "FergusBlind_Default": FergusBlindMotionDeblurFilter(fergus_cfg)
    }
    
    optuna_ssim_filter = get_filter_from_optuna(os.path.join(base_dir, "outputs", "motion_deblur", "optuna_config1_ssim", "best_params.json"))
    if optuna_ssim_filter:
        filters["Optuna_SSIM"] = optuna_ssim_filter
        
    optuna_brisque_filter = get_filter_from_optuna(os.path.join(base_dir, "outputs", "motion_deblur", "optuna_config2_brisque", "best_params.json"))
    if optuna_brisque_filter:
        filters["Optuna_BRISQUE"] = optuna_brisque_filter
    
    csv_file = os.path.join(base_dir, "outputs", "motion_deblur", "gopro_pairs.csv")
    dataset = GoProDataset(csv_file, split='test')
    
    output_dir = os.path.join(base_dir, "outputs", "motion_deblur", "visual_comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    num_visualizations = min(10, len(dataset))
    
    for i in range(num_visualizations):
        data = dataset[i]
        image_id = data['image_id']
        blur_img = data['blur']
        sharp_img = data['sharp']
        
        results = [("Blurry", blur_img), ("Sharp GT", sharp_img)]
        
        for name, filt in filters.items():
            restored = filt.apply(blur_img)
            results.append((name, restored))
            
        # Create figure
        fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
        for j, (title, img) in enumerate(results):
            # OpenCV uses BGR, convert to RGB for matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[j].imshow(img_rgb)
            axes[j].set_title(title)
            axes[j].axis("off")
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{image_id}_comparison.png"))
        plt.close(fig)
        print(f"Saved visualization for {image_id}")
        
    print("Visualizations complete.")

if __name__ == "__main__":
    main()
