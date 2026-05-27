import optuna
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.datasets.snow100k_dataset import Snow100kDataset
from src.optimization.desnowing.objective_snow100k_ssim import ObjectiveSnow100KSSIM

def main():
    study_name = "desnow_config1_ssim"
    os.makedirs("A:/HUST_on_GitHub/ProjectCV/results/optuna/desnow_config1_ssim", exist_ok=True)
    storage_path = f"sqlite:///A:/HUST_on_GitHub/ProjectCV/results/optuna/desnow_config1_ssim/study.db"
    
    val_file = "A:/HUST_on_GitHub/ProjectCV/data/snow100k/splits/val.txt"
    base_dir = "A:/HUST_on_GitHub/ProjectCV/dataset/Snow100K"
    dataset = Snow100kDataset(val_file, base_dir)
    
    objective = ObjectiveSnow100KSSIM(dataset)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # 50 trials for fast iteration, normally higher
    study.optimize(objective, n_trials=50)
    
    print("Best value: ", study.best_value)
    print("Best params: ", study.best_params)
    
    import yaml
    with open("A:/HUST_on_GitHub/ProjectCV/results/optuna/desnow_config1_ssim/best_config.yaml", "w") as f:
        yaml.dump(study.best_params, f)
        
    study.trials_dataframe().to_csv("A:/HUST_on_GitHub/ProjectCV/results/optuna/desnow_config1_ssim/trials.csv", index=False)

if __name__ == "__main__":
    main()
