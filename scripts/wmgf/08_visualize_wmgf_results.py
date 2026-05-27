import os
import cv2
import yaml
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.restoration.wmgf_derain import WMGFDerainFilter, WMGFConfig
from src.datasets.dawn_dataset import DawnDataset

def load_config(path, is_optuna=False):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return WMGFConfig(**cfg)

def main():
    configs = {
        "WMGF_Default": load_config("A:/HUST_on_GitHub/ProjectCV/configs/wmgf/derain_wmgf_default.yaml"),
        "WMGF_Config1": load_config("A:/HUST_on_GitHub/ProjectCV/results/optuna/wmgf_config1_ssim/best_config.yaml", True),
        "WMGF_Config2": load_config("A:/HUST_on_GitHub/ProjectCV/results/optuna/wmgf_config2_brisque/best_config.yaml", True),
        "WMGF_Config3": load_config("A:/HUST_on_GitHub/ProjectCV/results/optuna/wmgf_config3_map50/best_config.yaml", True)
    }
    
    dawn_val_file = "A:/HUST_on_GitHub/ProjectCV/data/dawn_rain/splits/rain_val_pairs.csv"
    if not os.path.exists(dawn_val_file):
        print(f"Data not found: {dawn_val_file}")
        return
        
    dataset = DawnDataset(dawn_val_file, "")
    
    vis_dir = "A:/HUST_on_GitHub/ProjectCV/results/visualizations/wmgf"
    os.makedirs(vis_dir, exist_ok=True)
    
    num_samples = min(5, len(dataset))
    
    for i in range(num_samples):
        sample = dataset[i]
        orig_img = sample["image"]
        basename = sample["image_rel"]
        
        cv2.imwrite(os.path.join(vis_dir, f"{i}_0_original_{basename}"), orig_img)
        
        for name, cfg in configs.items():
            if cfg is None:
                continue
            
            wmgf = WMGFDerainFilter(cfg)
            restored = wmgf.restore(orig_img)
            
            out_name = f"{i}_{name}_{basename}"
            cv2.imwrite(os.path.join(vis_dir, out_name), restored)
            
    print(f"Saved visualizations to {vis_dir}")

if __name__ == "__main__":
    main()
