import os
import glob
import random
import xml.etree.ElementTree as ET

def get_subset(pairs):
    total = len(pairs)
    if total > 10000:
        subset_size = max(int(0.1 * total), 1000)
        return random.sample(pairs, subset_size)
    return pairs

def split_and_save(pairs, split_dir):
    random.shuffle(pairs)
    split_idx = int(0.8 * len(pairs))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    os.makedirs(split_dir, exist_ok=True)
    with open(os.path.join(split_dir, "train_pairs.csv"), "w") as f:
        for img, lbl in train_pairs:
            f.write(f"{img},{lbl}\n")
    with open(os.path.join(split_dir, "val_test_pairs.csv"), "w") as f:
        for img, lbl in val_pairs:
            f.write(f"{img},{lbl}\n")
            
    print(f"Saved to {split_dir}: Train: {len(train_pairs)}, Val/Test: {len(val_pairs)}")

def process_foggy_cityscapes():
    print("Processing Foggy Cityscapes...")
    base_dir = "A:/HUST_on_GitHub/ProjectCV/dataset/Foggy_Cityscapes"
    images = glob.glob(os.path.join(base_dir, "**", "images", "*.jpg"), recursive=True) + \
             glob.glob(os.path.join(base_dir, "**", "images", "*.png"), recursive=True)
             
    pairs = []
    for img_path in images:
        lbl_path = img_path.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt")
        if os.path.exists(lbl_path):
            pairs.append((img_path.replace("\\", "/"), lbl_path.replace("\\", "/")))
            
    pairs = get_subset(pairs)
    split_dir = "A:/HUST_on_GitHub/ProjectCV/data/foggycityscapes/splits"
    split_and_save(pairs, split_dir)

def process_rtts():
    print("Processing RTTS...")
    base_dir = "A:/HUST_on_GitHub/ProjectCV/dataset/RTTS/RTTS"
    img_dir = os.path.join(base_dir, "JPEGImages")
    xml_dir = os.path.join(base_dir, "Annotations")
    lbl_dir = os.path.join(base_dir, "labels")
    os.makedirs(lbl_dir, exist_ok=True)
    
    class_map = {
        "person": 0,
        "bicycle": 1,
        "car": 2,
        "motorbike": 3,
        "bus": 5,
        "train": 6,
        "truck": 7
    }
    
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    pairs = []
    
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            size = root.find("size")
            width = float(size.find("width").text)
            height = float(size.find("height").text)
            
            if width == 0 or height == 0: continue
            
            yolo_lines = []
            for obj in root.findall("object"):
                cls_name = obj.find("name").text.lower()
                if cls_name not in class_map:
                    continue
                cls_id = class_map[cls_name]
                
                bndbox = obj.find("bndbox")
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
                
                cx = (xmin + xmax) / 2.0 / width
                cy = (ymin + ymax) / 2.0 / height
                bw = (xmax - xmin) / width
                bh = (ymax - ymin) / height
                
                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                
            img_name = os.path.basename(xml_file).replace(".xml", ".png")
            img_path = os.path.join(img_dir, img_name)
            
            if not os.path.exists(img_path):
                img_name = img_name.replace(".png", ".jpg")
                img_path = os.path.join(img_dir, img_name)
                
            if os.path.exists(img_path):
                lbl_path = os.path.join(lbl_dir, os.path.basename(xml_file).replace(".xml", ".txt"))
                with open(lbl_path, "w") as f:
                    f.write("\n".join(yolo_lines) + "\n")
                pairs.append((img_path.replace("\\", "/"), lbl_path.replace("\\", "/")))
        except Exception as e:
            pass
            
    pairs = get_subset(pairs)
    split_dir = "A:/HUST_on_GitHub/ProjectCV/data/rtts/splits"
    split_and_save(pairs, split_dir)

if __name__ == "__main__":
    random.seed(42)
    process_foggy_cityscapes()
    process_rtts()
