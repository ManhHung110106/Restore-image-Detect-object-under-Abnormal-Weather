import os
import cv2
import yaml
from pathlib import Path
from tqdm import tqdm
from src.restoration.lime_delowlight import LIMEDeLowlightConfig, LIMEDeLowlightFilter

def test_lime_default():
    config_path = "configs/lime_delowlight_default.yaml"
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
        
    # Remove 'name' since it's not in dataclass
    if 'name' in cfg_dict:
        del cfg_dict['name']
        
    cfg_dict['debug'] = True  # Enable debug mode for outputs
    config = LIMEDeLowlightConfig(**cfg_dict)
    
    lime_filter = LIMEDeLowlightFilter(config)
    
    val_low_dir = Path(r"data\lol\val\low")
    val_high_dir = Path(r"data\lol\val\high")
    
    debug_dir = Path(r"results\delowlight\debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    low_images = sorted(list(val_low_dir.glob("*.png")))
    num_to_save = 10
    
    for i, low_path in enumerate(tqdm(low_images)):
        img_bgr = cv2.imread(str(low_path))
        
        # Restore
        restored_bgr = lime_filter.restore(img_bgr)
        
        # Save debug outputs for first 10 images
        if i < num_to_save:
            img_dir = debug_dir / low_path.stem
            img_dir.mkdir(exist_ok=True)
            
            cv2.imwrite(str(img_dir / "input_low.png"), img_bgr)
            cv2.imwrite(str(img_dir / "illumination_initial.png"), lime_filter.debug_illumination_initial)
            cv2.imwrite(str(img_dir / "illumination_refined.png"), lime_filter.debug_illumination_refined)
            cv2.imwrite(str(img_dir / "enhanced_before_postprocess.png"), lime_filter.debug_enhanced_before_postprocess)
            cv2.imwrite(str(img_dir / "final_enhanced.png"), lime_filter.debug_final_enhanced)
            
            high_path = val_high_dir / low_path.name
            if high_path.exists():
                gt_bgr = cv2.imread(str(high_path))
                cv2.imwrite(str(img_dir / "gt_normal.png"), gt_bgr)

if __name__ == "__main__":
    test_lime_default()
