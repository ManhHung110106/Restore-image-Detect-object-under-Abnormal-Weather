import optuna
import yaml
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.datasets.dawn_dataset import DawnDataset
from src.detection.yolo_runner import YOLOEvaluator
from src.optimization.wmgf.objective_wmgf_config3_map50 import ObjectiveWMGFConfig3Map50

def main():
    with open("A:/HUST_on_GitHub/ProjectCV/configs/wmgf/derain_optuna_config3_map50.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    study_name = cfg["study_name"]
    storage_path = f"sqlite:///A:/HUST_on_GitHub/ProjectCV/results/optuna/wmgf_config3_map50/study.db"
    
    os.makedirs("A:/HUST_on_GitHub/ProjectCV/results/optuna/wmgf_config3_map50", exist_ok=True)
    
    val_file = "A:/HUST_on_GitHub/ProjectCV/data/dawn_rain/splits/rain_val_pairs.csv"
    dataset = DawnDataset(val_file, "")
    
    yolo_model_path = cfg["model"]["name"]
    yolo_eval = YOLOEvaluator(model_path=yolo_model_path, data_yaml="", device="auto")
    
    base_results_dir = "A:/HUST_on_GitHub/ProjectCV/results"
    
    objective = ObjectiveWMGFConfig3Map50(dataset, yolo_eval, base_results_dir)
    
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
    
    with open("A:/HUST_on_GitHub/ProjectCV/results/optuna/wmgf_config3_map50/best_config.yaml", "w") as f:
        yaml.dump(study.best_params, f)
        
    study.trials_dataframe().to_csv("A:/HUST_on_GitHub/ProjectCV/results/optuna/wmgf_config3_map50/trials.csv", index=False)

if __name__ == "__main__":
    main()
