import os
import json
import shutil
import random
from pathlib import Path

# Fix seed for reproducibility
random.seed(42)

def prepare_sots(source_dir, dest_dir, split_ratio=0.8):
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    
    # We will just create a JSON split file pointing to the absolute paths
    # to save space, or we can copy. Let's just create a JSON registry of pairs.
    
    pairs = []
    
    for environment in ['indoor', 'outdoor']:
        env_dir = source_dir / environment
        if not env_dir.exists():
            continue
            
        hazy_dir = env_dir / 'hazy'
        clear_dir = env_dir / 'clear'
        
        # In SOTS, often hazy images are named like '1400_1.png' while clear is '1400.png'
        # Or sometimes they have the exact same name if organized differently.
        # Let's list hazy images and find their clear counterparts.
        
        for hazy_file in hazy_dir.glob('*'):
            if not hazy_file.is_file():
                continue
                
            # SOTS naming convention:
            # Indoor: hazy: 1400_1.png -> clear: 1400.png
            # Outdoor: hazy: 0001_0.8_0.2.jpg -> clear: 0001.jpg or 0001.png
            name_parts = hazy_file.stem.split('_')
            base_name = name_parts[0]
            
            # Find matching clear file
            clear_candidates = [
                clear_dir / f"{base_name}.png",
                clear_dir / f"{base_name}.jpg",
                clear_dir / hazy_file.name # exact match
            ]
            
            clear_file = None
            for cand in clear_candidates:
                if cand.exists():
                    clear_file = cand
                    break
            
            if clear_file:
                pairs.append({
                    'hazy': str(hazy_file.absolute()),
                    'clear': str(clear_file.absolute()),
                    'env': environment
                })

    print(f"Found {len(pairs)} pairs in SOTS.")
    
    # Shuffle
    random.shuffle(pairs)
    
    split_idx = int(len(pairs) * split_ratio)
    tune_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]
    
    out_file = dest_dir / 'sots_split.json'
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_file, 'w') as f:
        json.dump({'tune': tune_pairs, 'test': test_pairs}, f, indent=4)
        
    print(f"Saved splits to {out_file}. Tune: {len(tune_pairs)}, Test: {len(test_pairs)}")

if __name__ == '__main__':
    # Using the kagglehub downloaded path
    kaggle_sots = r"C:\Users\manh hung\.cache\kagglehub\datasets\balraj98\synthetic-objective-testing-set-sots-reside\versions\1"
    out_dir = r"configs"
    prepare_sots(kaggle_sots, out_dir)
