import os
import sys
import cv2
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.restoration.lime_delowlight import LIMEDeLowlightConfig, LIMEDeLowlightFilter

def load_filter(config_path):
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    if 'name' in cfg_dict:
        del cfg_dict['name']
    config = LIMEDeLowlightConfig(**cfg_dict)
    return LIMEDeLowlightFilter(config)

def put_text(img, text):
    # Add text to image with background for visibility
    img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img, (0, 0), (w + 20, h + 20), (0, 0, 0), -1)
    cv2.putText(img, text, (10, h + 10), font, font_scale, (255, 255, 255), thickness)
    return img

def main():
    test_low_dir = Path("data/lol/test/low")
    test_high_dir = Path("data/lol/test/high")
    
    methods = {
        "Raw low-light": None,
        "GT": None,
        "LIME-default": load_filter("configs/lime_delowlight_default.yaml"),
        "LIME-config1-SSIM": load_filter("configs/lime_delowlight_config1_ssim.yaml"),
        "LIME-config2-BRISQUE": load_filter("configs/lime_delowlight_config2_brisque.yaml")
    }
    
    out_dir = Path("results/delowlight/figures/visual_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    test_low_images = sorted(list(test_low_dir.glob("*.png")))
    
    for i, low_path in enumerate(tqdm(test_low_images, desc="Visualizing")):
        high_path = test_high_dir / low_path.name
        if not high_path.exists():
            continue
            
        low_img = cv2.imread(str(low_path))
        high_img = cv2.imread(str(high_path))
        
        results = {}
        for name, filter_obj in methods.items():
            if name == "Raw low-light":
                res = low_img
            elif name == "GT":
                res = high_img
            else:
                if filter_obj is not None:
                    res = filter_obj.restore(low_img)
                else:
                    res = np.zeros_like(low_img)
            
            res_annotated = put_text(res, name)
            results[name] = res_annotated
            
        # Combine images side-by-side
        # Order: Low, GT, Default, Config1, Config2
        order = ["Raw low-light", "GT", "LIME-default", "LIME-config1-SSIM", "LIME-config2-BRISQUE"]
        images = [results[name] for name in order]
        
        combined = np.hstack(images)
        
        cv2.imwrite(str(out_dir / f"compare_{low_path.name}"), combined)
        
    print("Visual comparison completed.")

if __name__ == "__main__":
    main()
