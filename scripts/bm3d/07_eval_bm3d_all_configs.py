import os
import pandas as pd
import cv2
import json
from src.datasets.bsd_denoise_dataset import BSDDenoiseDataset
from src.restoration.registry import create_filter
from src.metrics.full_reference import compute_psnr, compute_ssim, compute_ms_ssim, compute_lpips, compute_mae, compute_mse
from src.metrics.no_reference import compute_brisque, compute_niqe, compute_piqe, compute_entropy

def eval_all():
    os.makedirs('results/metrics', exist_ok=True)
    
    configs = {
        'Raw noisy': None,
        'BM3D-default': 'configs/bm3d_default_noise{level}.yaml',
        'BM3D-config1-specialized': 'configs/bm3d_config1_bsd{level}_ssim.yaml',
        'BM3D-config2-specialized': 'configs/bm3d_config2_bsd{level}_brisque.yaml',
        'BM3D-config1-mixed': 'configs/bm3d_config1_bsd_mixed_ssim.yaml',
        'BM3D-config2-mixed': 'configs/bm3d_config2_bsd_mixed_brisque.yaml',
    }
    
    noise_levels = [15, 25, 50]
    
    results = []
    
    for level in noise_levels:
        dataset = BSDDenoiseDataset(root='data/bsd_denoise', noise_level=level, split='test')
        
        # We limit to 5 images for evaluation to save time for this scratch BM3D
        max_images = 5
        
        for method, cfg_path in configs.items():
            print(f"Evaluating {method} on Noise {level}")
            
            filt = None
            if cfg_path is not None:
                actual_path = cfg_path.format(level=level)
                if not os.path.exists(actual_path):
                    print(f"Skipping {method} for {level} because config {actual_path} is missing.")
                    continue
                filt = create_filter('bm3d', actual_path)
                
            for i in range(min(len(dataset), max_images)):
                noisy, clean, img_id = dataset[i]
                
                if filt is None:
                    # Raw noisy
                    restored = noisy
                else:
                    restored = filt.restore(noisy)
                    
                # Compute metrics
                res = {
                    'method': method,
                    'noise_level': level,
                    'image_id': img_id,
                    'PSNR': compute_psnr(restored, clean),
                    'SSIM': compute_ssim(restored, clean),
                    'MS-SSIM': compute_ms_ssim(restored, clean),
                    'LPIPS': compute_lpips(restored, clean),
                    'MAE': compute_mae(restored, clean),
                    'MSE': compute_mse(restored, clean),
                    'BRISQUE': compute_brisque(restored),
                    'NIQE': compute_niqe(restored),
                    'PIQE': compute_piqe(restored),
                    'Entropy': compute_entropy(restored)
                }
                results.append(res)
                
    df = pd.DataFrame(results)
    df.to_csv('results/metrics/denoise_all_metrics_summary.csv', index=False)
    
    # Save full reference separated
    df_fr = df[['method', 'noise_level', 'image_id', 'PSNR', 'SSIM', 'MS-SSIM', 'LPIPS', 'MAE', 'MSE']]
    df_fr.to_csv('results/metrics/denoise_full_reference_metrics.csv', index=False)
    
    # Save no reference separated
    df_nr = df[['method', 'noise_level', 'image_id', 'BRISQUE', 'NIQE', 'PIQE', 'Entropy']]
    df_nr.to_csv('results/metrics/denoise_no_reference_metrics.csv', index=False)
    
    # Save json summary (mean per method per level)
    summary = df.groupby(['method', 'noise_level']).mean(numeric_only=True).reset_index()
    summary_dict = summary.to_dict(orient='records')
    with open('results/metrics/denoise_all_metrics_summary.json', 'w') as f:
        json.dump(summary_dict, f, indent=4)
        
    print("Evaluation completed!")

if __name__ == "__main__":
    eval_all()
