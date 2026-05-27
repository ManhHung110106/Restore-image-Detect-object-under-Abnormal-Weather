import os
import glob
import random
import shutil

def main():
    base_dir = "A:/HUST_on_GitHub/ProjectCV/dataset/DAWN/Fog"
    yolo_labels_dir = os.path.join(base_dir, "Fog_YOLO_darknet")
    
    splits_dir = "A:/HUST_on_GitHub/ProjectCV/data/dawn/splits"
    os.makedirs(splits_dir, exist_ok=True)
    
    std_images_dir = "A:/HUST_on_GitHub/ProjectCV/data/dawn/images"
    std_labels_dir = "A:/HUST_on_GitHub/ProjectCV/data/dawn/labels"
    os.makedirs(std_images_dir, exist_ok=True)
    os.makedirs(std_labels_dir, exist_ok=True)
    
    images = sorted(glob.glob(os.path.join(base_dir, "*.jpg")))
    
    samples = []
    
    print("Standardizing DAWN YOLO format...")
    for img_path in images:
        basename = os.path.basename(img_path)
        label_name = os.path.splitext(basename)[0] + ".txt"
        label_path = os.path.join(yolo_labels_dir, label_name)
        
        # Output paths
        out_img_path = os.path.join(std_images_dir, basename)
        out_lbl_path = os.path.join(std_labels_dir, label_name)
        
        # Copy image if not exists
        if not os.path.exists(out_img_path):
            shutil.copy2(img_path, out_img_path)
            
        # Copy label if exists
        if os.path.exists(label_path):
            if not os.path.exists(out_lbl_path):
                shutil.copy2(label_path, out_lbl_path)
            samples.append((out_img_path, out_lbl_path))
        else:
            # Create empty label if no objects
            if not os.path.exists(out_lbl_path):
                open(out_lbl_path, 'w').close()
            samples.append((out_img_path, out_lbl_path))
            
    print(f"Found {len(samples)} fog images.")
    
    random.seed(42)
    random.shuffle(samples)
    
    val_count = int(len(samples) * 0.20)
    val_samples = samples[:val_count]
    test_samples = samples[val_count:]
    
    with open(os.path.join(splits_dir, "fog_val.txt"), "w") as f:
        for img, _ in val_samples:
            # write absolute path for yolo
            f.write(f"{img}\n")
            
    with open(os.path.join(splits_dir, "fog_test.txt"), "w") as f:
        for img, _ in test_samples:
            f.write(f"{img}\n")
            
    # Also save csv format for our DAWN Dataset loader
    with open(os.path.join(splits_dir, "fog_val_pairs.csv"), "w") as f:
        for img, lbl in val_samples:
            f.write(f"{img},{lbl}\n")
            
    with open(os.path.join(splits_dir, "fog_test_pairs.csv"), "w") as f:
        for img, lbl in test_samples:
            f.write(f"{img},{lbl}\n")
            
    print(f"Saved {len(val_samples)} to fog_val.txt and {len(test_samples)} to fog_test.txt")

if __name__ == "__main__":
    main()
