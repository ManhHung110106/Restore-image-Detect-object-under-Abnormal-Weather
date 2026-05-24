import os
import argparse
from pathlib import Path
import glob

import cv2
import optuna
import numpy as np
import torch

try:
    import pyiqa
except ImportError:
    print("Warning: pyiqa package not found. Please install with: pip install pyiqa")
    pyiqa = None

# Import our dehaze module
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.filters.dark_channel_prior import dehaze

# Initialize metrics if pyiqa is available
# Note: pyiqa returns tensors, usually we want to minimize NIQE and BRISQUE
# Smaller NIQE = better, Smaller BRISQUE = better
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if pyiqa is not None:
    niqe_metric = pyiqa.create_metric('niqe', as_loss=False).to(device)
    brisque_metric = pyiqa.create_metric('brisque', as_loss=False).to(device)
else:
    niqe_metric = None
    brisque_metric = None

def calc_no_ref_metrics(img_bgr):
    """Calculate NIQE and BRISQUE for a given BGR image in range [0, 255]"""
    if niqe_metric is None or brisque_metric is None:
        return 0.0, 0.0
        
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # pyiqa expects inputs as tensor [N, C, H, W] in range [0, 1]
    img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        niqe_score = niqe_metric(img_tensor).item()
        brisque_score = brisque_metric(img_tensor).item()
        
    return niqe_score, brisque_score

def objective(trial, image_paths, max_images=100):
    """
    Optuna objective function for multi-objective optimization (NIQE, BRISQUE).
    """
    # Suggest hyperparameters
    omega = trial.suggest_float("omega", 0.7, 0.99)
    t0 = trial.suggest_float("t0", 0.05, 0.2)
    patch_size = trial.suggest_int("patch_size", 5, 25, step=2)
    
    niqe_scores = []
    brisque_scores = []
    
    # Sample a subset of image_paths to save time during trials
    random_indices = np.random.choice(len(image_paths), min(max_images, len(image_paths)), replace=False)
    
    for i in random_indices:
        hazy_img = cv2.imread(image_paths[i])
        
        if hazy_img is None:
            continue
            
        # Dehaze
        dehazed = dehaze(hazy_img, patch_size=patch_size, omega=omega, t0=t0, refine=True)
        
        # Calculate metrics
        n_score, b_score = calc_no_ref_metrics(dehazed)
        
        niqe_scores.append(n_score)
        brisque_scores.append(b_score)
        
    avg_niqe = np.mean(niqe_scores) if niqe_scores else 100.0
    avg_brisque = np.mean(brisque_scores) if brisque_scores else 100.0
    
    # We want to MINIMIZE NIQE and MINIMIZE BRISQUE
    return avg_niqe, avg_brisque

def run_optuna_tuning(dawn_dir, n_trials=50, out_db="sqlite:///optuna_nogt.db", max_images=100):
    """
    Tune on the training and validation splits of DAWN.
    dawn_dir expects something like 'data/processed/dawn_yolo'
    """
    image_paths = []
    for split in ["train", "val"]:
        fog_dir = os.path.join(dawn_dir, "images", split, "Fog")
        if os.path.exists(fog_dir):
            for ext in ["*.jpg", "*.png", "*.jpeg"]:
                image_paths.extend(glob.glob(os.path.join(fog_dir, "**", ext), recursive=True))
            
    print(f"Starting Optuna tuning with {len(image_paths)} DAWN images available.")
    
    if len(image_paths) == 0:
        print("Error: No images found to tune on.")
        return

    # Multi-objective: minimize NIQE, minimize BRISQUE
    study = optuna.create_study(
        study_name="dcp_nogt_optimization",
        storage=out_db,
        load_if_exists=True,
        directions=["minimize", "minimize"]
    )
    
    study.optimize(lambda trial: objective(trial, image_paths, max_images=max_images), n_trials=n_trials)
    
    print("Number of finished trials: ", len(study.trials))
    print("Pareto front trials:")
    for t in study.best_trials:
        print(f"  Trial#{t.number} Values (NIQE, BRISQUE): {t.values} Params: {t.params}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dawn_dir", default="data/processed/dawn_yolo", help="Path to DAWN YOLO dataset")
    parser.add_argument("--trials", type=int, default=30, help="Number of trials")
    parser.add_argument("--out_db", default="sqlite:///optuna_nogt.db", help="Optuna database URI")
    parser.add_argument("--max_images", type=int, default=100, help="Max images per trial")
    args = parser.parse_args()
    
    run_optuna_tuning(args.dawn_dir, n_trials=args.trials, out_db=args.out_db, max_images=args.max_images)
