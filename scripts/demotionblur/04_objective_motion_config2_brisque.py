import os
import yaml
import cv2
import sys
import pandas as pd
import optuna
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.datasets.gopro_dataset import GoProDataset
from src.restoration.wiener_deblur import WienerMotionDeblurFilter
from src.restoration.richardson_lucy_deblur import RichardsonLucyMotionDeblurFilter
from src.restoration.fergus_blind_deblur import FergusBlindMotionDeblurFilter
from src.metrics.no_reference import compute_brisque

def objective(trial, dataset):
    method = trial.suggest_categorical("method", ["wiener", "richardson_lucy", "fergus_blind"])
    
    if method == "wiener":
        kernel_length = trial.suggest_int("kernel_length", 3, 45, step=2)
        kernel_angle = trial.suggest_float("kernel_angle", 0.0, 180.0)
        reg_lambda = trial.suggest_float("regularization_lambda", 1e-6, 1e-1, log=True)
        post_sharpen_amount = trial.suggest_float("post_sharpen_amount", 0.0, 1.0)
        
        cfg = {
            'kernel_length': kernel_length,
            'kernel_angle': kernel_angle,
            'regularization_lambda': reg_lambda,
            'post_sharpen': True if post_sharpen_amount > 0 else False,
            'post_sharpen_amount': post_sharpen_amount,
            'clip_output': True
        }
        filt = WienerMotionDeblurFilter(cfg)
        
    elif method == "richardson_lucy":
        kernel_length = trial.suggest_int("kernel_length", 3, 45, step=2)
        kernel_angle = trial.suggest_float("kernel_angle", 0.0, 180.0)
        num_iter = trial.suggest_int("num_iter", 5, 80)
        denoise_after = trial.suggest_categorical("denoise_after", [False, True])
        
        cfg = {
            'kernel_length': kernel_length,
            'kernel_angle': kernel_angle,
            'num_iter': num_iter,
            'denoise_after': denoise_after,
            'clip_output': True
        }
        filt = RichardsonLucyMotionDeblurFilter(cfg)
        
    elif method == "fergus_blind":
        kernel_length = trial.suggest_int("kernel_length", 5, 65, step=2)
        kernel_angle = trial.suggest_float("kernel_angle", 0.0, 180.0)
        deconv_method = trial.suggest_categorical("deconv_method", ["wiener", "richardson_lucy"])
        reg_lambda = trial.suggest_float("regularization_lambda", 1e-6, 1e-1, log=True)
        num_iter = trial.suggest_int("num_iter", 10, 80)
        
        cfg = {
            'per_image_search': False,
            'kernel_length': kernel_length,
            'kernel_angle': kernel_angle,
            'deconv_method': deconv_method,
            'regularization_lambda': reg_lambda,
            'num_iter': num_iter,
            'clip_output': True
        }
        filt = FergusBlindMotionDeblurFilter(cfg)
        
    brisque_scores = []
    # Evaluate on a subset of validation (e.g. 50 images to speed up search)
    subset_size = min(50, len(dataset))
    for i in range(subset_size):
        data = dataset[i]
        restored = filt.apply(data['blur'])
        # compute BRISQUE using the function that handles uint8 BGR
        score = compute_brisque(restored)
        if not np.isnan(score):
            brisque_scores.append(score)
            
    if len(brisque_scores) == 0:
        return float('inf')
        
    mean_brisque = sum(brisque_scores) / len(brisque_scores)
    return mean_brisque

def main():
    base_dir = r"A:\HUST_on_GitHub\ProjectCV"
    csv_file = os.path.join(base_dir, "outputs", "motion_deblur", "gopro_pairs.csv")
    dataset = GoProDataset(csv_file, split='val')
    
    output_dir = os.path.join(base_dir, "outputs", "motion_deblur", "optuna_config2_brisque")
    os.makedirs(output_dir, exist_ok=True)
    
    study = optuna.create_study(direction="minimize", study_name="GoPro_BRISQUE")
    
    # Fast evaluation to test pipeline. n_trials=50.
    study.optimize(lambda trial: objective(trial, dataset), n_trials=50)
    
    print("Best params:", study.best_params)
    print("Best BRISQUE:", study.best_value)
    
    # Save best params
    import json
    with open(os.path.join(output_dir, "best_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=4)
        
    # Save trials to csv
    study.trials_dataframe().to_csv(os.path.join(output_dir, "trials.csv"))

if __name__ == "__main__":
    main()
