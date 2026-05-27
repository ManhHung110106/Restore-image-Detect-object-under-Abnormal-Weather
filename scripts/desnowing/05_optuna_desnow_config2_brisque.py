import optuna
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.datasets.dawn_dataset import DawnDataset
from src.optimization.desnowing.objective_dawn_brisque import ObjectiveDawnBrisque

def main():
    study_name = "desnow_config2_brisque"
    os.makedirs("A:/HUST_on_GitHub/ProjectCV/results/optuna/desnow_config2_brisque", exist_ok=True)
    storage_path = f"sqlite:///A:/HUST_on_GitHub/ProjectCV/results/optuna/desnow_config2_brisque/study.db"
    
    val_file = "A:/HUST_on_GitHub/ProjectCV/data/dawn/splits/snow_val_pairs.csv"
    dataset = DawnDataset(val_file, "")
    
    objective = ObjectiveDawnBrisque(dataset)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=50)
    
    print("Best value: ", study.best_value)
    print("Best params: ", study.best_params)
    
    import yaml
    with open("A:/HUST_on_GitHub/ProjectCV/results/optuna/desnow_config2_brisque/best_config.yaml", "w") as f:
        yaml.dump(study.best_params, f)
        
    study.trials_dataframe().to_csv("A:/HUST_on_GitHub/ProjectCV/results/optuna/desnow_config2_brisque/trials.csv", index=False)

if __name__ == "__main__":
    main()
