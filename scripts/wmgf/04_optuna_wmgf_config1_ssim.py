import optuna
import yaml
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.datasets.rain100H_dataset import PairedRainDataset
from src.optimization.wmgf.objective_wmgf_config1_ssim import ObjectiveWMGFConfig1SSIM

def main():
    with open("A:/HUST_on_GitHub/ProjectCV/configs/wmgf/derain_optuna_config1_ssim.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    study_name = cfg["study_name"]
    storage_path = f"sqlite:///A:/HUST_on_GitHub/ProjectCV/results/optuna/wmgf_config1_ssim/study.db"
    
    os.makedirs("A:/HUST_on_GitHub/ProjectCV/results/optuna/wmgf_config1_ssim", exist_ok=True)
    
    val_file = "A:/HUST_on_GitHub/ProjectCV/data/rain100h/splits/val.txt"
    base_dir = "A:/HUST_on_GitHub/ProjectCV/dataset/rain100H"
    dataset = PairedRainDataset(val_file, base_dir)
    
    objective = ObjectiveWMGFConfig1SSIM(dataset)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        direction=cfg["direction"],
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=cfg["seed"])
    )
    
    study.optimize(objective, n_trials=cfg["n_trials"])
    
    print("Best value: ", study.best_value)
    print("Best params: ", study.best_params)
    
    with open("A:/HUST_on_GitHub/ProjectCV/results/optuna/wmgf_config1_ssim/best_config.yaml", "w") as f:
        yaml.dump(study.best_params, f)
        
    study.trials_dataframe().to_csv("A:/HUST_on_GitHub/ProjectCV/results/optuna/wmgf_config1_ssim/trials.csv", index=False)

if __name__ == "__main__":
    main()
