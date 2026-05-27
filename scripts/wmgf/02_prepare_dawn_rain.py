import os
import glob
import random
import shutil

def main():
    base_dir = r"A:\HUST_on_GitHub\ProjectCV\dataset\DAWN\Rain"
    yolo_labels_dir = os.path.join(base_dir, "Rain_YOLO_darknet")
    
    splits_dir = r"A:\HUST_on_GitHub\ProjectCV\data\dawn_rain\splits"

    os.makedirs(splits_dir, exist_ok=True)
    
    std_images_dir = r"A:\HUST_on_GitHub\ProjectCV\data\dawn_rain\images"
    std_labels_dir = r"A:\HUST_on_GitHub\ProjectCV\data\dawn_rain\labels"
    os.makedirs(std_images_dir, exist_ok=True)
    os.makedirs(std_labels_dir, exist_ok=True)
    
    images = sorted(glob.glob(os.path.join(base_dir, "*.jpg")))
    
    samples = []
    
    print("Standardizing DAWN Rain YOLO format...")
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
            # DAWN Rain uses 1-indexed COCO classes (1=person, 3=car, 4=motorcycle, 6=bus, 8=truck)
            # We must subtract 1 to get standard 0-indexed COCO classes expected by YOLOv26
            with open(label_path, "r") as f_in, open(out_lbl_path, "w") as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        parts[0] = str(class_id - 1)
                        f_out.write(" ".join(parts) + "\n")
            samples.append((out_img_path, out_lbl_path))
        else:
            # Create empty label if no objects
            if not os.path.exists(out_lbl_path):
                open(out_lbl_path, 'w').close()
            samples.append((out_img_path, out_lbl_path))
            
    print(f"Found {len(samples)} rain images.")
    
    random.seed(42)
    random.shuffle(samples)
    
    val_count = int(len(samples) * 0.20)
    val_samples = samples[:val_count]
    test_samples = samples[val_count:]
    
    with open(os.path.join(splits_dir, "rain_val.txt"), "w") as f:
        for img, _ in val_samples:
            # write absolute path for yolo
            f.write(f"{img}\n")
            
    with open(os.path.join(splits_dir, "rain_test.txt"), "w") as f:
        for img, _ in test_samples:
            f.write(f"{img}\n")
            
    # Also save csv format for our DAWN Dataset loader
    with open(os.path.join(splits_dir, "rain_val_pairs.csv"), "w") as f:
        for img, lbl in val_samples:
            f.write(f"{img},{lbl}\n")
            
    with open(os.path.join(splits_dir, "rain_test_pairs.csv"), "w") as f:
        for img, lbl in test_samples:
            f.write(f"{img},{lbl}\n")
            
    print(f"Saved {len(val_samples)} to rain_val.txt and {len(test_samples)} to rain_test.txt")

if __name__ == "__main__":
    main()
