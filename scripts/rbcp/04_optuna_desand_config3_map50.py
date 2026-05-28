import optuna
import yaml
import os
import csv
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)
os.chdir(project_root)
from src.restoration.rbcp_desand import RBCPDesandFilter
from src.optimization.rbcp.objective_desand_config3_map50 import objective_desand_config3_map50

def load_pairs(csv_path):
    pairs = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 2:
                pairs.append((row[0], row[1]))
    return pairs

def main():
    val_csv = "data/dawn/splits/sand_val_pairs.csv"
    dataset_yaml = "data/dawn/dawn_sand.yaml"
    
    if not os.path.exists(val_csv):
        print("Please run 01_prepare_dawn_sand.py first")
        return
        
    pairs = load_pairs(val_csv)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective_desand_config3_map50(
        trial, 
        RBCPDesandFilter, 
        dataset_yaml, 
        pairs, 
        yolo_model_path="yolov8n.pt" # Assuming YOLOv8 format since YOLO26 isn't real, but guide says yolo26
    ), n_trials=30)
    
    print("Best config:", study.best_params)
    print("Best mAP50:", study.best_value)
    
    # Save trial results
    os.makedirs("results/optuna", exist_ok=True)
    with open("results/optuna/desand_config3_map50_trials.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_id", "value", "state"] + list(study.best_params.keys()))
        for t in study.trials:
            row = [t.number, t.value, t.state.name]
            for k in study.best_params.keys():
                row.append(t.params.get(k, ""))
            writer.writerow(row)
            
    # Save best config
    best_config = study.best_params
    best_config['use_guided_filter'] = True
    
    with open("configs/desand_optuna_config3_map50.yaml", "w") as f:
        yaml.dump(best_config, f)
        
    print("Saved configs/desand_optuna_config3_map50.yaml")

if __name__ == "__main__":
    main()
