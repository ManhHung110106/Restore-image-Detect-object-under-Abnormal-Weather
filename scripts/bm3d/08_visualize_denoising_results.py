import os
import cv2
import numpy as np
from src.datasets.bsd_denoise_dataset import BSDDenoiseDataset
from src.restoration.registry import create_filter

def visualize():
    os.makedirs('results/figures/denoise_visual_comparison', exist_ok=True)
    
    configs = {
        'Raw noisy': None,
        'BM3D-default': 'configs/bm3d_default_noise{level}.yaml',
        'BM3D-config1-specialized': 'configs/bm3d_config1_bsd{level}_ssim.yaml',
        'BM3D-config2-specialized': 'configs/bm3d_config2_bsd{level}_brisque.yaml',
        'BM3D-config1-mixed': 'configs/bm3d_config1_bsd_mixed_ssim.yaml',
        'BM3D-config2-mixed': 'configs/bm3d_config2_bsd_mixed_brisque.yaml',
    }
    
    noise_levels = [15, 25, 50]
    
    for level in noise_levels:
        out_dir = f'results/figures/denoise_visual_comparison/noise{level}'
        os.makedirs(out_dir, exist_ok=True)
        
        dataset = BSDDenoiseDataset(root='data/bsd_denoise', noise_level=level, split='test')
        
        # Take 1 image for visualization to save time
        noisy, clean, img_id = dataset[0]
        
        cv2.imwrite(os.path.join(out_dir, f'{img_id}_clean.png'), clean)
        cv2.imwrite(os.path.join(out_dir, f'{img_id}_raw_noisy.png'), noisy)
        
        for method, cfg_path in configs.items():
            if cfg_path is None:
                continue
                
            actual_path = cfg_path.format(level=level)
            if not os.path.exists(actual_path):
                continue
                
            filt = create_filter('bm3d', actual_path)
            restored = filt.restore(noisy)
            
            safe_method_name = method.replace(' ', '_').replace('-', '_')
            cv2.imwrite(os.path.join(out_dir, f'{img_id}_{safe_method_name}.png'), restored)
            
        print(f"Visualizations saved for noise {level}")

if __name__ == "__main__":
    visualize()
