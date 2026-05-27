import optuna
import yaml
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.datasets.dawn_dataset import DawnDataset
from src.optimization.dcp.objective_dawn_brisque import ObjectiveDawnBrisque

def main():
    with open("A:/HUST_on_GitHub/ProjectCV/configs/dcp/optuna_config2_dawn_brisque.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    study_name = cfg["study_name"]
    storage_path = f"sqlite:///A:/HUST_on_GitHub/ProjectCV/results/optuna/config2_dawn_brisque/study.db"
    
    os.makedirs("A:/HUST_on_GitHub/ProjectCV/results/optuna/config2_dawn_brisque", exist_ok=True)
    
    # Load validation split
    dawn_val_file = "A:/HUST_on_GitHub/ProjectCV/data/dawn/splits/fog_val_pairs.csv"
    dataset = DawnDataset(dawn_val_file, "")
    
    objective = ObjectiveDawnBrisque(dataset)
    
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
    
    with open("A:/HUST_on_GitHub/ProjectCV/results/optuna/config2_dawn_brisque/best_config.yaml", "w") as f:
        yaml.dump(study.best_params, f)
        
    study.trials_dataframe().to_csv("A:/HUST_on_GitHub/ProjectCV/results/optuna/config2_dawn_brisque/trials.csv", index=False)

if __name__ == "__main__":
    main()
