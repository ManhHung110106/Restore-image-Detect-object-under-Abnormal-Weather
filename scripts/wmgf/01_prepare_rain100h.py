import os
import glob
import random

def main():
    base_dir = "A:/HUST_on_GitHub/ProjectCV/dataset/rain100H"
    train_dir = os.path.join(base_dir, "train")
    rainy_dir = os.path.join(train_dir, "rain")
    clean_dir = os.path.join(train_dir, "norain")
    
    splits_dir = "A:/HUST_on_GitHub/ProjectCV/data/rain100h/splits"
    os.makedirs(splits_dir, exist_ok=True)
    
    rainy_images = sorted(glob.glob(os.path.join(rainy_dir, "*.jpg")) + glob.glob(os.path.join(rainy_dir, "*.png")))
    
    pairs = []
    for rainy_path in rainy_images:
        basename = os.path.basename(rainy_path)
        # Assuming matching filenames
        # Example in rain100H: rainy name `norain-1x2.png` -> wait, we assume name matches
        # We will match exact basename. Sometimes suffix needs adjusting, e.g. norain-xxx -> rain-xxx.
        clean_path = os.path.join(clean_dir, basename)
        
        # Check if exists, else try replacing 'rain' with 'norain' if it differs
        if not os.path.exists(clean_path):
            alt_basename = basename.replace("rain", "norain") if "norain" not in basename else basename.replace("norain", "rain")
            clean_path = os.path.join(clean_dir, alt_basename)
            if not os.path.exists(clean_path):
                # Another try: maybe the 'norain' file just has the ID
                import re
                m = re.search(r'\d+', basename)
                if m:
                    idx = m.group(0)
                    for ext in [".png", ".jpg"]:
                        if os.path.exists(os.path.join(clean_dir, f"{idx}{ext}")):
                            clean_path = os.path.join(clean_dir, f"{idx}{ext}")
                            break
                        if os.path.exists(os.path.join(clean_dir, f"norain-{idx}{ext}")):
                            clean_path = os.path.join(clean_dir, f"norain-{idx}{ext}")
                            break

        if os.path.exists(clean_path):
            r_rel = os.path.relpath(rainy_path, base_dir).replace('\\', '/')
            c_rel = os.path.relpath(clean_path, base_dir).replace('\\', '/')
            pairs.append((r_rel, c_rel))
        else:
            print(f"Warning: Missing clean image for {rainy_path}")
            
    print(f"Found {len(pairs)} pairs in rain100H test folder.")
    
    # Shuffle and split 20% val, 80% test (since we only have test dir here, we split it for optuna/eval)
    random.seed(42)
    random.shuffle(pairs)
    
    val_count = int(len(pairs) * 0.20)
    val_pairs = pairs[:val_count]
    test_pairs = pairs[val_count:]
    
    with open(os.path.join(splits_dir, "val.txt"), "w") as f:
        for r, c in val_pairs:
            f.write(f"{r},{c}\n")
            
    with open(os.path.join(splits_dir, "test.txt"), "w") as f:
        for r, c in test_pairs:
            f.write(f"{r},{c}\n")
            
    print(f"Saved {len(val_pairs)} to val.txt and {len(test_pairs)} to test.txt")

if __name__ == "__main__":
    main()
