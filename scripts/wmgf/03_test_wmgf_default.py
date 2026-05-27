import os
import yaml
import cv2
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.restoration.wmgf_derain import WMGFConfig, WMGFDerainFilter

def main():
    config_path = "A:/HUST_on_GitHub/ProjectCV/configs/wmgf/derain_wmgf_default.yaml"
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
        
    config = WMGFConfig(**cfg_dict)
    wmgf = WMGFDerainFilter(config)
    
    print("WMGF Filter initialized successfully.")
    
    # Try with a random dummy image
    dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    print("Running restoration on dummy image...")
    res_img, debug = wmgf.restore(dummy_img, return_debug=True)
    print("Restoration completed.")
    print("Output shape:", res_img.shape)
    for k, v in debug.items():
        print(f"Debug {k}: shape {v.shape}")

if __name__ == "__main__":
    main()
