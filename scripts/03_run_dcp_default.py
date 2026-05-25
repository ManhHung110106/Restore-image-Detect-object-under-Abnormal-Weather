import os
import yaml
import cv2
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.restoration.dcp import DCPConfig, DCPDehazeFilter
from src.datasets.reside_dataset import ResideDataset
from src.datasets.dawn_dataset import DawnDataset

def main():
    config_path = "A:/HUST_on_GitHub/ProjectCV/configs/dcp_default.yaml"
    with open(config_path, "r") as f:
        d_cfg = yaml.safe_load(f)
        
    config = DCPConfig(
        patch_size=d_cfg.get("patch_size", 15),
        omega=d_cfg.get("omega", 0.95),
        t0=d_cfg.get("t0", 0.10),
        top_percent=d_cfg.get("top_percent", 0.001),
        guided_radius=d_cfg.get("guided_radius", 40),
        guided_eps=d_cfg.get("guided_eps", 0.001),
        gamma=d_cfg.get("gamma", 1.0),
        clahe_clip=d_cfg.get("clahe_clip", 0.0),
    )
    
    dcp = DCPDehazeFilter(config)
    
    # Process Reside
    reside_test_file = "A:/HUST_on_GitHub/ProjectCV/data/reside6k/splits/test.txt"
    reside_base_dir = "A:/HUST_on_GitHub/ProjectCV/dataset/RESIDE-6K"
    if os.path.exists(reside_test_file):
        print("Processing RESIDE Test...")
        reside_dataset = ResideDataset(reside_test_file, reside_base_dir)
        out_dir = "A:/HUST_on_GitHub/ProjectCV/results/restored/dcp_default/reside_test"
        os.makedirs(out_dir, exist_ok=True)
        
        # Process a few samples to save debug info
        for i, sample in enumerate(reside_dataset):
            res_img, debug = dcp.restore(sample["hazy"], return_debug=True)
            basename = os.path.basename(sample["hazy_path"])
            cv2.imwrite(os.path.join(out_dir, basename), res_img)
            
            if i < 5:
                debug_dir = f"A:/HUST_on_GitHub/ProjectCV/results/figures/debug_dcp_default/reside/{i}"
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, "dark.jpg"), debug["dark_channel"])
                cv2.imwrite(os.path.join(debug_dir, "t_raw.jpg"), debug["transmission_raw"])
                cv2.imwrite(os.path.join(debug_dir, "t_ref.jpg"), debug["transmission_refined"])
                cv2.imwrite(os.path.join(debug_dir, "res.jpg"), res_img)

    # Process DAWN
    dawn_test_file = "A:/HUST_on_GitHub/ProjectCV/data/dawn/splits/fog_test.txt"
    dawn_base_dir = "A:/HUST_on_GitHub/ProjectCV/dataset/DAWN/Fog"
    # Actually, we use fog_test_pairs.csv which has absolute paths. Or we can just read fog_test.txt.
    dawn_test_pairs = "A:/HUST_on_GitHub/ProjectCV/data/dawn/splits/fog_test_pairs.csv"
    if os.path.exists(dawn_test_pairs):
        print("Processing DAWN Fog Test...")
        # Empty base dir because the csv contains absolute paths
        dawn_dataset = DawnDataset(dawn_test_pairs, "")
        out_dir = "A:/HUST_on_GitHub/ProjectCV/results/restored/dcp_default/dawn_fog_test"
        os.makedirs(out_dir, exist_ok=True)
        
        for i, sample in enumerate(dawn_dataset):
            res_img, debug = dcp.restore(sample["image"], return_debug=True)
            basename = sample["image_rel"]
            cv2.imwrite(os.path.join(out_dir, basename), res_img)
            
            if i < 5:
                debug_dir = f"A:/HUST_on_GitHub/ProjectCV/results/figures/debug_dcp_default/dawn/{i}"
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, "dark.jpg"), debug["dark_channel"])
                cv2.imwrite(os.path.join(debug_dir, "t_raw.jpg"), debug["transmission_raw"])
                cv2.imwrite(os.path.join(debug_dir, "t_ref.jpg"), debug["transmission_refined"])
                cv2.imwrite(os.path.join(debug_dir, "res.jpg"), res_img)

if __name__ == "__main__":
    main()
