import optuna
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.datasets.dawn_dataset import DawnDataset
from src.detection.yolo_runner import YOLOEvaluator
from src.optimization.desnowing.objective_dawn_map50 import ObjectiveDawnMap50

def main():
    study_name = "desnow_config3_map50"
    os.makedirs("A:/HUST_on_GitHub/ProjectCV/results/optuna/desnow_config3_map50", exist_ok=True)
    storage_path = f"sqlite:///A:/HUST_on_GitHub/ProjectCV/results/optuna/desnow_config3_map50/study.db"
    
    val_file = "A:/HUST_on_GitHub/ProjectCV/data/dawn/splits/snow_val_pairs.csv"
    dataset = DawnDataset(val_file, "")
    
    yolo_eval = YOLOEvaluator(model_path="A:/HUST_on_GitHub/ProjectCV/yolo26n.pt", data_yaml="", device="auto")
    
    base_results_dir = "A:/HUST_on_GitHub/ProjectCV/results"
    
    objective = ObjectiveDawnMap50(dataset, yolo_eval, base_results_dir)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=30) # fewer trials for object detection to save time
    
    print("Best value: ", study.best_value)
    print("Best params: ", study.best_params)
    
    import yaml
    with open("A:/HUST_on_GitHub/ProjectCV/results/optuna/desnow_config3_map50/best_config.yaml", "w") as f:
        yaml.dump(study.best_params, f)
        
    study.trials_dataframe().to_csv("A:/HUST_on_GitHub/ProjectCV/results/optuna/desnow_config3_map50/trials.csv", index=False)

if __name__ == "__main__":
    main()
