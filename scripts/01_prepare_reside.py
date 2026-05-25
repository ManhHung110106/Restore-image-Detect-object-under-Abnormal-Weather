import os
import glob
import random

def main():
    base_dir = "A:/HUST_on_GitHub/ProjectCV/dataset/RESIDE-6K"
    test_dir = os.path.join(base_dir, "test")
    hazy_dir = os.path.join(test_dir, "hazy")
    gt_dir = os.path.join(test_dir, "GT")
    
    splits_dir = "A:/HUST_on_GitHub/ProjectCV/data/reside6k/splits"
    os.makedirs(splits_dir, exist_ok=True)
    
    hazy_images = sorted(glob.glob(os.path.join(hazy_dir, "*.jpg")) + glob.glob(os.path.join(hazy_dir, "*.png")))
    
    pairs = []
    for hazy_path in hazy_images:
        basename = os.path.basename(hazy_path)
        gt_path = os.path.join(gt_dir, basename)
        
        if os.path.exists(gt_path):
            # store relative paths to base_dir
            h_rel = os.path.relpath(hazy_path, base_dir)
            g_rel = os.path.relpath(gt_path, base_dir)
            # Use forward slashes
            h_rel = h_rel.replace('\\', '/')
            g_rel = g_rel.replace('\\', '/')
            pairs.append((h_rel, g_rel))
        else:
            print(f"Warning: Missing GT for {hazy_path}")
            
    print(f"Found {len(pairs)} pairs in RESIDE test folder.")
    
    # Shuffle and split 20% val, 80% test
    random.seed(42)
    random.shuffle(pairs)
    
    val_count = int(len(pairs) * 0.20)
    val_pairs = pairs[:val_count]
    test_pairs = pairs[val_count:]
    
    with open(os.path.join(splits_dir, "val.txt"), "w") as f:
        for h, c in val_pairs:
            f.write(f"{h},{c}\n")
            
    with open(os.path.join(splits_dir, "test.txt"), "w") as f:
        for h, c in test_pairs:
            f.write(f"{h},{c}\n")
            
    print(f"Saved {len(val_pairs)} to val.txt and {len(test_pairs)} to test.txt")

if __name__ == "__main__":
    main()
