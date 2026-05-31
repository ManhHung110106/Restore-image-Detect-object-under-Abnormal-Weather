import os
import yaml
import cv2
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.datasets.gopro_dataset import GoProDataset
from src.restoration.wiener_deblur import WienerMotionDeblurFilter
from src.restoration.richardson_lucy_deblur import RichardsonLucyMotionDeblurFilter
from src.restoration.fergus_blind_deblur import FergusBlindMotionDeblurFilter

from src.metrics.full_reference import compute_psnr, compute_ssim, compute_ms_ssim, compute_lpips, compute_mae, compute_mse
from src.metrics.no_reference import compute_brisque, compute_niqe, compute_piqe, compute_entropy

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_metrics(restored, sharp):
    metrics = {}
    metrics['PSNR'] = compute_psnr(restored, sharp)
    metrics['SSIM'] = compute_ssim(restored, sharp)
    metrics['MS-SSIM'] = compute_ms_ssim(restored, sharp)
    metrics['LPIPS'] = compute_lpips(restored, sharp)
    metrics['MAE'] = compute_mae(restored, sharp)
    metrics['MSE'] = compute_mse(restored, sharp)
    
    metrics['BRISQUE'] = compute_brisque(restored)
    metrics['NIQE'] = compute_niqe(restored)
    metrics['PIQE'] = compute_piqe(restored)
    metrics['Entropy'] = compute_entropy(restored)
    return metrics

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
    
    # 1. Load default configs
    wiener_cfg = load_config(os.path.join(base_dir, "configs", "motion_deblur", "wiener_default.yaml"))
    rl_cfg = load_config(os.path.join(base_dir, "configs", "motion_deblur", "richardson_lucy_default.yaml"))
    fergus_cfg = load_config(os.path.join(base_dir, "configs", "motion_deblur", "fergus_blind_default.yaml"))
    
    # Initialize filters
    filters = {
        "Raw_Blurry": None,
        "Wiener_Default": WienerMotionDeblurFilter(wiener_cfg),
        "RichardsonLucy_Default": RichardsonLucyMotionDeblurFilter(rl_cfg),
        "FergusBlind_Default": FergusBlindMotionDeblurFilter(fergus_cfg)
    }
    
    # Add Optuna filters if available
    optuna_ssim_filter = get_filter_from_optuna(os.path.join(base_dir, "outputs", "motion_deblur", "optuna_config1_ssim", "best_params.json"))
    if optuna_ssim_filter:
        filters["Optuna_SSIM"] = optuna_ssim_filter
        
    optuna_brisque_filter = get_filter_from_optuna(os.path.join(base_dir, "outputs", "motion_deblur", "optuna_config2_brisque", "best_params.json"))
    if optuna_brisque_filter:
        filters["Optuna_BRISQUE"] = optuna_brisque_filter
    
    csv_file = os.path.join(base_dir, "outputs", "motion_deblur", "gopro_pairs.csv")
    dataset = GoProDataset(csv_file, split='test')
    
    output_dir = os.path.join(base_dir, "outputs", "motion_deblur", "final_eval")
    os.makedirs(output_dir, exist_ok=True)
    
    # For speed in demonstration, we can evaluate on a subset of the test set
    eval_size = min(20, len(dataset))
    
    results = []
    
    for i in range(eval_size):
        data = dataset[i]
        image_id = data['image_id']
        blur_img = data['blur']
        sharp_img = data['sharp']
        
        for name, filt in filters.items():
            if filt is None:
                restored = blur_img
            else:
                restored = filt.apply(blur_img)
                
            metrics = run_metrics(restored, sharp_img)
            metrics['Image_ID'] = image_id
            metrics['Method'] = name
            results.append(metrics)
            print(f"Evaluated {name} on {image_id}")
            
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "metrics_per_image.csv"), index=False)
    
    # Summary
    summary = df.drop(columns=['Image_ID']).groupby('Method').mean()
    summary.to_csv(os.path.join(output_dir, "metrics_summary.csv"))
    print("Final evaluation completed.")
    print(summary)

if __name__ == "__main__":
    main()
