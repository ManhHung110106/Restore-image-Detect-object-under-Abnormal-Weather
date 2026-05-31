import os
import cv2
import yaml
from src.datasets.bsd_denoise_dataset import BSDDenoiseDataset
from src.restoration.registry import create_filter

def test_default():
    print("Testing BM3D Default Configs")
    
    noise_levels = [15, 25, 50]
    for level in noise_levels:
        print(f"\n--- Noise {level} ---")
        dataset = BSDDenoiseDataset(root='data/bsd_denoise', noise_level=level, split='test')
        config_path = f'configs/bm3d_default_noise{level}.yaml'
        
        filt = create_filter('bm3d', config_path)
        
        # Test on 1 image
        noisy, clean, img_id = dataset[0]
        
        print(f"Processing image {img_id}")
        denoised = filt.restore(noisy)
        
        print("Done!")
        
if __name__ == "__main__":
    test_default()
