import os
import sys
import cv2
import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.restoration.lime_delowlight import LIMEDeLowlightConfig, LIMEDeLowlightFilter
from src.metrics.full_reference import (
    compute_psnr, compute_ssim, compute_ms_ssim, 
    compute_lpips, compute_mae, compute_mse
)
from src.metrics.no_reference import (
    compute_brisque, compute_niqe, compute_piqe, compute_entropy
)

def load_filter(config_path):
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    if 'name' in cfg_dict:
        del cfg_dict['name']
    config = LIMEDeLowlightConfig(**cfg_dict)
    return LIMEDeLowlightFilter(config)

def main():
    test_low_dir = Path("data/lol/test/low")
    test_high_dir = Path("data/lol/test/high")
    
    methods = {
        "Raw low-light": None,
        "LIME-default": load_filter("configs/lime_delowlight_default.yaml"),
        "LIME-config1-LOL-SSIM": load_filter("configs/lime_delowlight_config1_ssim.yaml"),
        "LIME-config2-LOL-BRISQUE": load_filter("configs/lime_delowlight_config2_brisque.yaml")
    }
    
    # Check if all models are loaded
    for name, f in methods.items():
        if name != "Raw low-light" and f is None:
            print(f"Warning: Configuration for {name} not found. Ensure Optuna scripts were run.")
            return

    test_low_images = sorted(list(test_low_dir.glob("*.png")))
    
    results = []
    
    for low_path in tqdm(test_low_images, desc="Evaluating"):
        high_path = test_high_dir / low_path.name
        if not high_path.exists():
            continue
            
        low_img = cv2.imread(str(low_path))
        high_img = cv2.imread(str(high_path))
        
        for method_name, filter_obj in methods.items():
            if method_name == "Raw low-light":
                restored = low_img
            else:
                restored = filter_obj.restore(low_img)
                
            # Full reference metrics
            psnr = compute_psnr(restored, high_img)
            ssim = compute_ssim(restored, high_img)
            ms_ssim = compute_ms_ssim(restored, high_img)
            lpips_val = compute_lpips(restored, high_img)
            mae = compute_mae(restored, high_img)
            mse = compute_mse(restored, high_img)
            
            # No reference metrics
            brisque = compute_brisque(restored)
            niqe = compute_niqe(restored)
            piqe = compute_piqe(restored)
            entropy = compute_entropy(restored)
            
            results.append({
                "Method": method_name,
                "Image": low_path.name,
                "PSNR": psnr,
                "SSIM": ssim,
                "MS-SSIM": ms_ssim,
                "LPIPS": lpips_val,
                "MAE": mae,
                "MSE": mse,
                "BRISQUE": brisque,
                "NIQE": niqe,
                "PIQE": piqe,
                "Entropy": entropy
            })
            
    df = pd.DataFrame(results)
    
    # Calculate means per method
    summary_df = df.groupby("Method").mean(numeric_only=True).reset_index()
    
    # Reorder methods for presentation
    method_order = ["Raw low-light", "LIME-default", "LIME-config1-LOL-SSIM", "LIME-config2-LOL-BRISQUE"]
    summary_df["Method"] = pd.Categorical(summary_df["Method"], categories=method_order, ordered=True)
    summary_df = summary_df.sort_values("Method")
    
    out_dir = Path("results/delowlight/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Full reference summary
    fr_cols = ["Method", "PSNR", "SSIM", "MS-SSIM", "LPIPS", "MAE", "MSE"]
    fr_df = summary_df[fr_cols]
    fr_df.to_csv(out_dir / "lime_full_reference_metrics.csv", index=False)
    
    # No reference summary
    nr_cols = ["Method", "BRISQUE", "NIQE", "PIQE", "Entropy"]
    nr_df = summary_df[nr_cols]
    nr_df.to_csv(out_dir / "lime_no_reference_metrics.csv", index=False)
    
    # All metrics summary
    summary_df.to_csv(out_dir / "lime_all_metrics_summary.csv", index=False)
    
    # Save to JSON
    summary_dict = summary_df.set_index("Method").to_dict(orient="index")
    with open(out_dir / "lime_all_metrics_summary.json", "w") as f:
        json.dump(summary_dict, f, indent=4)
        
    print("Evaluation completed. Results saved to results/delowlight/metrics/")

if __name__ == "__main__":
    main()
