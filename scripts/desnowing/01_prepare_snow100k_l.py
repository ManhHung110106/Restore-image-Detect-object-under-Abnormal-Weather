import os
import glob
import random

def main():
    base_dir = "A:/HUST_on_GitHub/ProjectCV/dataset/Snow100K"
    synthetic_dir = os.path.join(base_dir, "synthetic")
    gt_dir = os.path.join(base_dir, "gt")
    
    splits_dir = "A:/HUST_on_GitHub/ProjectCV/data/snow100k/splits"
    os.makedirs(splits_dir, exist_ok=True)
    
    synthetic_images = sorted(glob.glob(os.path.join(synthetic_dir, "*.jpg")) + glob.glob(os.path.join(synthetic_dir, "*.png")))
    
    pairs = []
    for syn_path in synthetic_images:
        basename = os.path.basename(syn_path)
        gt_path = os.path.join(gt_dir, basename)
        
        if os.path.exists(gt_path):
            syn_rel = os.path.relpath(syn_path, base_dir).replace('\\', '/')
            gt_rel = os.path.relpath(gt_path, base_dir).replace('\\', '/')
            pairs.append((syn_rel, gt_rel))
            
    print(f"Found {len(pairs)} pairs in Snow100K.")
    
    # Shuffle and split 20% val, 80% test
    random.seed(42)
    random.shuffle(pairs)
    
    # Because it is a huge dataset, we only use the first 1/10 of the dataset
    sub_data = pairs[:len(pairs)//10]  # 1/10 of the dataset

    val_count = int(len(sub_data) * 0.20)
    val_pairs = sub_data[:val_count]
    test_pairs = sub_data[val_count:]
    
    with open(os.path.join(splits_dir, "val.txt"), "w") as f:
        for s, c in val_pairs:
            f.write(f"{s},{c}\n")
            
    with open(os.path.join(splits_dir, "test.txt"), "w") as f:
        for s, c in test_pairs:
            f.write(f"{s},{c}\n")
            
    print(f"Saved {len(val_pairs)} to val.txt and {len(test_pairs)} to test.txt")

if __name__ == "__main__":
    main()
