import os
import sys
import cv2
import yaml
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.restoration.lime_delowlight import LIMEDeLowlightConfig, LIMEDeLowlightFilter
from src.metrics.full_reference import compute_ssim

def sample_lime_config(trial):
    return LIMEDeLowlightConfig(
        illumination_floor=trial.suggest_float("illumination_floor", 0.01, 0.12),
        illumination_power=trial.suggest_float("illumination_power", 0.45, 1.10),
        
        guided_radius=trial.suggest_int("guided_radius", 4, 48),
        guided_eps=trial.suggest_float("guided_eps", 1e-5, 1e-1, log=True),
        
        exposure_gain=trial.suggest_float("exposure_gain", 0.8, 1.8),
        gamma=trial.suggest_float("gamma", 0.7, 1.8),
        blend_alpha=trial.suggest_float("blend_alpha", 0.5, 1.0),
        
        use_clahe=trial.suggest_categorical("use_clahe", [False, True]),
        clahe_clip=trial.suggest_float("clahe_clip", 0.0, 3.0),
        clahe_tile_grid_size=trial.suggest_categorical("clahe_tile_grid_size", [4, 8, 12]),
        
        use_denoise=trial.suggest_categorical("use_denoise", [False, True]),
        denoise_h=trial.suggest_float("denoise_h", 1.0, 8.0),
        denoise_h_color=trial.suggest_float("denoise_h_color", 1.0, 8.0),
        denoise_template_window=trial.suggest_categorical("denoise_template_window", [5, 7]),
        denoise_search_window=trial.suggest_categorical("denoise_search_window", [15, 21, 31]),
    )

def main():
    val_low_dir = Path("data/lol/val/low")
    val_high_dir = Path("data/lol/val/high")
    
    # Load all images into memory to speed up optuna trials
    print("Loading images into memory...")
    low_images = sorted(list(val_low_dir.glob("*.png")))
    dataset = []
    for low_path in low_images:
        high_path = val_high_dir / low_path.name
        if high_path.exists():
            low_img = cv2.imread(str(low_path))
            high_img = cv2.imread(str(high_path))
            dataset.append((low_img, high_img))
    
    print(f"Loaded {len(dataset)} validation pairs.")

    def objective_config1_ssim(trial):
        config = sample_lime_config(trial)
        lime_filter = LIMEDeLowlightFilter(config)
        
        scores = []
        for low_img, high_gt in dataset:
            restored = lime_filter.restore(low_img)
            score = compute_ssim(restored, high_gt)
            scores.append(score)
            
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", study_name="lime_delowlight_ssim")
    study.optimize(objective_config1_ssim, n_trials=20)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Save best config
    best_config_dict = trial.params
    best_config_dict["name"] = "lime_delowlight_config1_ssim"
    
    # We must ensure all fields are set, even defaults that optuna didn't pick
    best_params = trial.params.copy()
    if 'name' in best_params:
        del best_params['name']
    best_config_obj = LIMEDeLowlightConfig(**best_params)
    final_dict = {
        "name": "lime_delowlight_config1_ssim",
        "illumination_floor": best_config_obj.illumination_floor,
        "illumination_power": best_config_obj.illumination_power,
        "guided_radius": best_config_obj.guided_radius,
        "guided_eps": best_config_obj.guided_eps,
        "exposure_gain": best_config_obj.exposure_gain,
        "gamma": best_config_obj.gamma,
        "blend_alpha": best_config_obj.blend_alpha,
        "use_clahe": best_config_obj.use_clahe,
        "clahe_clip": best_config_obj.clahe_clip,
        "clahe_tile_grid_size": best_config_obj.clahe_tile_grid_size,
        "use_denoise": best_config_obj.use_denoise,
        "denoise_h": best_config_obj.denoise_h,
        "denoise_h_color": best_config_obj.denoise_h_color,
        "denoise_template_window": best_config_obj.denoise_template_window,
        "denoise_search_window": best_config_obj.denoise_search_window,
    }
    
    config_path = Path("configs/lime_delowlight_config1_ssim.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(final_dict, f, default_flow_style=False, sort_keys=False)
        
    # Save trial results
    results_dir = Path("results/delowlight/optuna")
    results_dir.mkdir(parents=True, exist_ok=True)
    df = study.trials_dataframe()
    df.to_csv(results_dir / "lime_config1_ssim_trials.csv", index=False)
    print("Optuna optimization completed and saved.")

if __name__ == "__main__":
    main()
