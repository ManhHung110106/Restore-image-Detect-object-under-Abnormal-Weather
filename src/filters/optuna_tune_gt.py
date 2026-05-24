import os
import json
import argparse
from pathlib import Path

import cv2
import optuna
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import torch

# Assuming lpips is installed. If not, we need to handle it or instruct the user to install it.
try:
    import lpips
    # Initialize LPIPS model once
    lpips_model = lpips.LPIPS(net='alex')
    if torch.cuda.is_available():
        lpips_model = lpips_model.cuda()
except ImportError:
    print("Warning: lpips package not found. Please install with: pip install lpips")
    lpips_model = None

# Import our dehaze module
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.filters.dark_channel_prior import dehaze

def calc_lpips(img1_bgr, img2_bgr):
    """Calculate LPIPS between two BGR images in range [0, 255]"""
    if lpips_model is None:
        return 0.0
    
    # Convert BGR to RGB
    img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)
    
    # LPIPS expects images in range [-1, 1] as tensor [N, C, H, W]
    img1_tensor = torch.from_numpy(img1_rgb).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    img2_tensor = torch.from_numpy(img2_rgb).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    
    if torch.cuda.is_available():
        img1_tensor = img1_tensor.cuda()
        img2_tensor = img2_tensor.cuda()
        
    with torch.no_grad():
        dist = lpips_model(img1_tensor, img2_tensor)
        
    return dist.item()


def objective(trial, tune_pairs, reside_root, max_images=100):
    """
    Optuna objective function for multi-objective optimization (PSNR, SSIM, LPIPS).
    """
    # Suggest hyperparameters
    omega = trial.suggest_float("omega", 0.7, 0.99)
    t0 = trial.suggest_float("t0", 0.05, 0.2)
    patch_size = trial.suggest_int("patch_size", 5, 25, step=2)
    
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    
    # Sample a subset of tune_pairs if it's too large, or run on all
    random_indices = np.random.choice(len(tune_pairs), min(max_images, len(tune_pairs)), replace=False)
    
    for i in random_indices:
        pair = tune_pairs[i]
        hazy_path = pair['hazy']
        clear_path = pair['clear']
        
        # Resolve relative paths
        if not os.path.isabs(hazy_path) and reside_root:
            hazy_path = os.path.join(reside_root, hazy_path)
        if not os.path.isabs(clear_path) and reside_root:
            clear_path = os.path.join(reside_root, clear_path)
            
        hazy_img = cv2.imread(hazy_path)
        clear_img = cv2.imread(clear_path)
        
        if hazy_img is None or clear_img is None:
            continue
            
        # Ensure same size just in case
        if hazy_img.shape != clear_img.shape:
            h, w = clear_img.shape[:2]
            hazy_img = cv2.resize(hazy_img, (w, h))
            
        # Dehaze
        dehazed = dehaze(hazy_img, patch_size=patch_size, omega=omega, t0=t0, refine=True)
        
        # Calculate metrics
        p = psnr(clear_img, dehazed)
        s = ssim(clear_img, dehazed, channel_axis=2)
        l = calc_lpips(dehazed, clear_img)
        
        psnr_scores.append(p)
        ssim_scores.append(s)
        lpips_scores.append(l)
        
    avg_psnr = np.mean(psnr_scores) if psnr_scores else 0.0
    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0.0
    avg_lpips = np.mean(lpips_scores) if lpips_scores else 1.0 # 1.0 is bad LPIPS
    
    # We want to MAXIMIZE psnr and ssim, and MINIMIZE lpips
    return avg_psnr, avg_ssim, avg_lpips

def run_optuna_tuning(config_path, reside_root, n_trials=50, out_db="sqlite:///optuna_gt.db", max_images=100):
    # Load dataset pairs
    with open(config_path, 'r') as f:
        splits = json.load(f)
    tune_pairs = splits.get('tune', [])
    
    print(f"Starting Optuna tuning with {len(tune_pairs)} pairs available.")
    
    # Multi-objective: maximize PSNR, maximize SSIM, minimize LPIPS
    study = optuna.create_study(
        study_name="dcp_gt_optimization",
        storage=out_db,
        load_if_exists=True,
        directions=["maximize", "maximize", "minimize"]
    )
    
    study.optimize(lambda trial: objective(trial, tune_pairs, reside_root, max_images=max_images), n_trials=n_trials)
    
    print("Number of finished trials: ", len(study.trials))
    print("Pareto front trials:")
    for t in study.best_trials:
        print(f"  Trial#{t.number} Values (PSNR, SSIM, LPIPS): {t.values} Params: {t.params}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_file", default="configs/reside6k.json", help="Path to JSON split file")
    parser.add_argument("--reside_root", default=r"C:\Users\manh hung\.cache\kagglehub\datasets\kmljts\reside-6k\versions\1\RESIDE-6K", help="Root of RESIDE dataset")
    parser.add_argument("--trials", type=int, default=30, help="Number of trials")
    parser.add_argument("--out_db", default="sqlite:///optuna_gt.db", help="Optuna database URI")
    parser.add_argument("--max_images", type=int, default=100, help="Max images per trial")
    args = parser.parse_args()
    
    run_optuna_tuning(args.split_file, args.reside_root, n_trials=args.trials, out_db=args.out_db, max_images=args.max_images)
