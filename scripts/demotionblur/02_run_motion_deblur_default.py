import os
import yaml
import cv2
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.datasets.gopro_dataset import GoProDataset
from src.restoration.wiener_deblur import WienerMotionDeblurFilter
from src.restoration.richardson_lucy_deblur import RichardsonLucyMotionDeblurFilter
from src.restoration.fergus_blind_deblur import FergusBlindMotionDeblurFilter

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    base_dir = r"A:\HUST_on_GitHub\ProjectCV"
    wiener_cfg = load_config(os.path.join(base_dir, "configs", "motion_deblur", "wiener_default.yaml"))
    rl_cfg = load_config(os.path.join(base_dir, "configs", "motion_deblur", "richardson_lucy_default.yaml"))
    fergus_cfg = load_config(os.path.join(base_dir, "configs", "motion_deblur", "fergus_blind_default.yaml"))
    
    wiener_filter = WienerMotionDeblurFilter(wiener_cfg)
    rl_filter = RichardsonLucyMotionDeblurFilter(rl_cfg)
    fergus_filter = FergusBlindMotionDeblurFilter(fergus_cfg)
    
    csv_file = os.path.join(base_dir, "outputs", "motion_deblur", "gopro_pairs.csv")
    dataset = GoProDataset(csv_file, split='val')
    
    output_dir = os.path.join(base_dir, "outputs", "motion_deblur", "default")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running default configs on first 5 val images...")
    
    for i in range(min(5, len(dataset))):
        data = dataset[i]
        image_id = data['image_id']
        blur_img = data['blur']
        sharp_img = data['sharp']
        
        # apply filters
        wiener_out = wiener_filter.apply(blur_img)
        rl_out = rl_filter.apply(blur_img)
        fergus_out = fergus_filter.apply(blur_img)
        
        # save
        cv2.imwrite(os.path.join(output_dir, f"{image_id}_blur.png"), blur_img)
        cv2.imwrite(os.path.join(output_dir, f"{image_id}_gt.png"), sharp_img)
        cv2.imwrite(os.path.join(output_dir, f"{image_id}_wiener.png"), wiener_out)
        cv2.imwrite(os.path.join(output_dir, f"{image_id}_rl.png"), rl_out)
        cv2.imwrite(os.path.join(output_dir, f"{image_id}_fergus.png"), fergus_out)
        
        print(f"Processed {image_id}")
        
    print("Done.")

if __name__ == "__main__":
    main()
