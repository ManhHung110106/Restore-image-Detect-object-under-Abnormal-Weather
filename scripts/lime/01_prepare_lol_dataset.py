import os
import shutil
import random
from pathlib import Path

def prepare_lol_dataset():
    source_dir = Path(r"A:\HUST_on_GitHub\ProjectCV\dataset\LOL")
    dest_dir = Path(r"A:\HUST_on_GitHub\ProjectCV\data\lol")

    our485_dir = source_dir / "our485"
    eval15_dir = source_dir / "eval15"

    if not our485_dir.exists() or not eval15_dir.exists():
        print("LOL dataset not found at source directory. Please check the path.")
        return

    # Create destination directories
    for split in ["train", "val", "test"]:
        for light in ["high", "low"]:
            (dest_dir / split / light).mkdir(parents=True, exist_ok=True)

    # Prepare test split from eval15
    print("Preparing test split...")
    for light in ["high", "low"]:
        src_files = list((eval15_dir / light).glob("*.png"))
        for f in src_files:
            shutil.copy2(f, dest_dir / "test" / light / f.name)

    # Prepare train and val splits from our485
    # Use a fixed seed for reproducibility
    random.seed(42)
    our485_files = sorted([f.name for f in (our485_dir / "high").glob("*.png")])
    
    # Shuffle and split: 400 for train, 85 for val
    random.shuffle(our485_files)
    train_files = our485_files[:400]
    val_files = our485_files[400:]

    print(f"Preparing train split ({len(train_files)} images)...")
    for f_name in train_files:
        shutil.copy2(our485_dir / "high" / f_name, dest_dir / "train" / "high" / f_name)
        shutil.copy2(our485_dir / "low" / f_name, dest_dir / "train" / "low" / f_name)

    print(f"Preparing val split ({len(val_files)} images)...")
    for f_name in val_files:
        shutil.copy2(our485_dir / "high" / f_name, dest_dir / "val" / "high" / f_name)
        shutil.copy2(our485_dir / "low" / f_name, dest_dir / "val" / "low" / f_name)

    print("LOL dataset preparation complete.")

if __name__ == "__main__":
    prepare_lol_dataset()
