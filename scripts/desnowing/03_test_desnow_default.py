import os
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.restoration.registry import create_filter

def main():
    config_path = "A:/HUST_on_GitHub/ProjectCV/configs/desnow_morph_guided_default.yaml"
    filter_obj = create_filter("desnow", config_path)
    
    test_img_path = "A:/HUST_on_GitHub/ProjectCV/data/dawn/images/snow/032486926.jpg" # Example dawn snow image
    # Let's just pick any image from data/dawn/images/snow/
    import glob
    images = glob.glob("A:/HUST_on_GitHub/ProjectCV/data/dawn/images/snow/*.jpg")
    if not images:
        print("No images found for testing.")
        return
        
    test_img_path = images[0]
    print(f"Testing on {test_img_path}")
    
    img = cv2.imread(test_img_path)
    if img is None:
        print("Failed to read image.")
        return
        
    restored, debug_info = filter_obj.restore(img, debug=True)
    
    out_dir = "A:/HUST_on_GitHub/ProjectCV/results/desnow_debug"
    os.makedirs(out_dir, exist_ok=True)
    
    basename = os.path.basename(test_img_path)
    cv2.imwrite(os.path.join(out_dir, f"original_{basename}"), img)
    cv2.imwrite(os.path.join(out_dir, f"restored_{basename}"), restored)
    
    for k, v in debug_info.items():
        cv2.imwrite(os.path.join(out_dir, f"{k}_{basename}"), v)
        
    print(f"Debug images saved to {out_dir}")

if __name__ == "__main__":
    main()
