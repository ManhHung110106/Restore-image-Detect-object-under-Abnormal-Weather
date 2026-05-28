import os
import cv2
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)
os.chdir(project_root)
from src.restoration.registry import create_filter

def main():
    config_path = "configs/desand_rbcp_default.yaml"
    filter_instance = create_filter("desand", config_path)
    
    # Take a few images from the test set
    test_split = "data/dawn/splits/test_sand.txt"
    if not os.path.exists(test_split):
        print("Please run 01_prepare_dawn_sand.py first")
        return
        
    with open(test_split, "r") as f:
        images = [line.strip() for line in f.readlines()]
        
    import random
    random.seed(42)
    sample_images = random.sample(images, min(10, len(images)))
    
    out_dir = "results/figures/desand_visual_comparison/debug_default"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Testing RBCP default on {len(sample_images)} images...")
    
    for i, img_path in enumerate(sample_images):
        basename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None: continue
            
        print(f"Processing {basename}...")
        debug_sub = os.path.join(out_dir, f"{i:02d}_{os.path.splitext(basename)[0]}")
        os.makedirs(debug_sub, exist_ok=True)
        
        filter_instance.restore(img, debug_dir=debug_sub)
        
    print("Done! Check debug outputs in:", out_dir)

if __name__ == "__main__":
    main()
