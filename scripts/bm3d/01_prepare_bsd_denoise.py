import os
import shutil
import random

def prepare_dataset():
    random.seed(42)
    
    src_dir = os.path.abspath('dataset/BSD')
    dst_dir = os.path.abspath('data/bsd_denoise')
    
    noise_levels = [15, 25, 50]
    
    # We will compute the split using the noise15 files, then apply same split for all
    # Because BSD typically uses the same set of images with different noise.
    sample_dir = os.path.join(src_dir, 'BSD_noisy15', 'input')
    if not os.path.exists(sample_dir):
        print(f"Directory not found: {sample_dir}")
        return
        
    all_files = sorted(os.listdir(sample_dir))
    random.shuffle(all_files)
    
    num_val = int(len(all_files) * 0.20)
    val_files = all_files[:num_val]
    test_files = all_files[num_val:]
    
    # Generate splits globally (or locally inside data/bsd_denoise/)
    os.makedirs(dst_dir, exist_ok=True)
    
    with open(os.path.join(dst_dir, 'val.txt'), 'w') as f:
        for fname in val_files:
            f.write(fname + '\n')
            
    with open(os.path.join(dst_dir, 'test.txt'), 'w') as f:
        for fname in test_files:
            f.write(fname + '\n')
            
    print(f"Generated splits: {len(val_files)} for val, {len(test_files)} for test.")
    
    for level in noise_levels:
        src_noise = os.path.join(src_dir, f'BSD_noisy{level}')
        dst_noise = os.path.join(dst_dir, f'noise{level}')
        
        dst_noisy_dir = os.path.join(dst_noise, 'noisy')
        dst_clean_dir = os.path.join(dst_noise, 'clean')
        
        os.makedirs(dst_noisy_dir, exist_ok=True)
        os.makedirs(dst_clean_dir, exist_ok=True)
        
        # Copy or symlink files
        for fname in all_files:
            # Input
            src_in = os.path.join(src_noise, 'input', fname)
            dst_in = os.path.join(dst_noisy_dir, fname)
            if not os.path.exists(dst_in):
                shutil.copy2(src_in, dst_in)
                
            # Restored (Clean)
            src_gt = os.path.join(src_noise, 'restored', fname)
            dst_gt = os.path.join(dst_clean_dir, fname)
            if not os.path.exists(dst_gt):
                shutil.copy2(src_gt, dst_gt)
                
        print(f"Prepared noise{level} dataset.")
        
if __name__ == "__main__":
    prepare_dataset()
