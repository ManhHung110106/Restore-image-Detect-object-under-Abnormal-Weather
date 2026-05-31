import os
import optuna
import pandas as pd
import yaml
from src.datasets.bsd_denoise_dataset import BSDDenoiseDataset
from src.optimization.bm3d.objective_bm3d_config2_specialized_brisque import objective_config2_specialized_brisque

def run_optuna_specialized_brisque():
    os.makedirs('results/optuna', exist_ok=True)
    os.makedirs('configs', exist_ok=True)
    
    noise_levels = [15, 25, 50]
    
    for level in noise_levels:
        print(f"Running Optuna for Config2 Specialized BRISQUE - Noise {level}")
        val_dataset = BSDDenoiseDataset(root='data/bsd_denoise', noise_level=level, split='val')
        
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective_config2_specialized_brisque(trial, level, val_dataset), n_trials=20)
        
        # Save trials
        df = study.trials_dataframe()
        df.to_csv(f'results/optuna/bm3d_config2_bsd{level}_brisque_trials.csv', index=False)
        
        # Save best config
        best_params = study.best_params
        config_data = {
            'name': f'bm3d_config2_bsd{level}_brisque',
            'sigma_psd': best_params['sigma_psd'],
            'stage_arg': best_params['stage_arg'],
            'profile': 'default'
        }
        with open(f'configs/bm3d_config2_bsd{level}_brisque.yaml', 'w') as f:
            yaml.dump(config_data, f)
            
        print(f"Best params for {level}: {best_params}")

if __name__ == "__main__":
    run_optuna_specialized_brisque()
