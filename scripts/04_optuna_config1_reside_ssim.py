import optuna
import yaml
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.datasets.reside_dataset import ResideDataset
from src.optimization.objective_reside_ssim import ObjectiveResideSSIM

def main():
    with open("A:/HUST_on_GitHub/ProjectCV/configs/optuna_config1_reside_ssim.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    study_name = cfg["study_name"]
    storage_path = f"sqlite:///A:/HUST_on_GitHub/ProjectCV/results/optuna/config1_reside_ssim/study.db"
    
    os.makedirs("A:/HUST_on_GitHub/ProjectCV/results/optuna/config1_reside_ssim", exist_ok=True)
    
    # Load validation split
    reside_val_file = "A:/HUST_on_GitHub/ProjectCV/data/reside6k/splits/val.txt"
    reside_base_dir = "A:/HUST_on_GitHub/ProjectCV/dataset/RESIDE-6K"
    dataset = ResideDataset(reside_val_file, reside_base_dir)
    
    objective = ObjectiveResideSSIM(dataset)
    
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
    
    # Save best config
    with open("A:/HUST_on_GitHub/ProjectCV/results/optuna/config1_reside_ssim/best_config.yaml", "w") as f:
        yaml.dump(study.best_params, f)
        
    # Save trials as CSV
    study.trials_dataframe().to_csv("A:/HUST_on_GitHub/ProjectCV/results/optuna/config1_reside_ssim/trials.csv", index=False)

if __name__ == "__main__":
    main()
