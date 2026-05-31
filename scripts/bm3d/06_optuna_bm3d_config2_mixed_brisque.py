import os
import optuna
import pandas as pd
import yaml
from src.datasets.bsd_denoise_dataset import BSDDenoiseDataset
from src.optimization.bm3d.objective_bm3d_config2_mixed_brisque import objective_config2_mixed_brisque

def run_optuna_mixed_brisque():
    os.makedirs('results/optuna', exist_ok=True)
    os.makedirs('configs', exist_ok=True)
    
    val_datasets = {
        15: BSDDenoiseDataset(root='data/bsd_denoise', noise_level=15, split='val'),
        25: BSDDenoiseDataset(root='data/bsd_denoise', noise_level=25, split='val'),
        50: BSDDenoiseDataset(root='data/bsd_denoise', noise_level=50, split='val')
    }
    
    print(f"Running Optuna for Config2 Mixed BRISQUE")
    
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective_config2_mixed_brisque(trial, val_datasets), n_trials=20)
    
    # Save trials
    df = study.trials_dataframe()
    df.to_csv(f'results/optuna/bm3d_config2_bsd_mixed_brisque_trials.csv', index=False)
    
    # Save best config
    best_params = study.best_params
    config_data = {
        'name': 'bm3d_config2_bsd_mixed_brisque',
        'sigma_psd': best_params['sigma_psd'],
        'stage_arg': best_params['stage_arg'],
        'profile': 'default'
    }
    with open('configs/bm3d_config2_bsd_mixed_brisque.yaml', 'w') as f:
        yaml.dump(config_data, f)
        
    print(f"Best params for mixed: {best_params}")

if __name__ == "__main__":
    run_optuna_mixed_brisque()
