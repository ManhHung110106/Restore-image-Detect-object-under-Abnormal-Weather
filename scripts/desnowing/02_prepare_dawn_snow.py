import os
import glob
import random
import shutil
import xml.etree.ElementTree as ET

COCO_CLASS_TO_ID = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "train": 6,
    "truck": 7
}
CLASS_TO_ID = COCO_CLASS_TO_ID

def convert_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def main():
    base_dataset_dir = "A:/HUST_on_GitHub/ProjectCV/dataset/DAWN/Snow"
    xml_dir = os.path.join(base_dataset_dir, "Snow_PASCAL_VOC")
    
    out_base_dir = "A:/HUST_on_GitHub/ProjectCV/data/dawn"
    out_images_dir = os.path.join(out_base_dir, "images", "snow")
    out_labels_dir = os.path.join(out_base_dir, "labels_yolo", "snow")
    splits_dir = os.path.join(out_base_dir, "splits")
    
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)
    
    # Also find corresponding images, we assume they are in base_dataset_dir directly
    # Wait, earlier list_dir showed images in dataset/DAWN/Snow/
    images = glob.glob(os.path.join(base_dataset_dir, "*.jpg"))
    
    samples = []
    
    for img_path in images:
        basename = os.path.basename(img_path)
        img_name, ext = os.path.splitext(basename)
        xml_path = os.path.join(xml_dir, img_name + ".xml")
        
        out_img_path = os.path.join(out_images_dir, basename)
        out_lbl_path = os.path.join(out_labels_dir, img_name + ".txt")
        
        if not os.path.exists(out_img_path):
            shutil.copy2(img_path, out_img_path)
            
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            
            with open(out_lbl_path, "w") as f_out:
                for obj in root.iter('object'):
                    cls = obj.find('name').text
                    if cls not in CLASS_TO_ID:
                        print(f"Warning: Unknown class '{cls}' in {xml_path}")
                        continue
                    cls_id = CLASS_TO_ID[cls]
                    xmlbox = obj.find('bndbox')
                    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                    bb = convert_to_yolo((w, h), b)
                    f_out.write(f"{cls_id} {' '.join(str(a) for a in bb)}\n")
                    
            samples.append((out_img_path, out_lbl_path))
        else:
            # Create empty label if no objects
            if not os.path.exists(out_lbl_path):
                open(out_lbl_path, 'w').close()
            samples.append((out_img_path, out_lbl_path))
            
    print(f"Prepared {len(samples)} DAWN Snow images.")
    
    random.seed(42)
    random.shuffle(samples)
    
    val_count = int(len(samples) * 0.20)
    val_samples = samples[:val_count]
    test_samples = samples[val_count:]
    
    # For YOLO
    with open(os.path.join(splits_dir, "val_snow.txt"), "w") as f:
        for img, _ in val_samples:
            f.write(f"{img}\n")
            
    with open(os.path.join(splits_dir, "test_snow.txt"), "w") as f:
        for img, _ in test_samples:
            f.write(f"{img}\n")
            
    # For DAWN dataset loader
    with open(os.path.join(splits_dir, "snow_val_pairs.csv"), "w") as f:
        for img, lbl in val_samples:
            f.write(f"{img},{lbl}\n")
            
    with open(os.path.join(splits_dir, "snow_test_pairs.csv"), "w") as f:
        for img, lbl in test_samples:
            f.write(f"{img},{lbl}\n")
            
    print(f"Saved {len(val_samples)} to val_snow.txt and {len(test_samples)} to test_snow.txt")

    # Generate dawn_snow.yaml
    yaml_path = os.path.join(out_base_dir, "dawn_snow.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {out_base_dir}\n")
        f.write("train: splits/train_snow.txt\n") # train might not exist, YOLO will warn but it's fine if we just run eval
        f.write("val: splits/val_snow.txt\n")
        f.write("test: splits/test_snow.txt\n\n")
        f.write("names:\n")
        for k, v in CLASS_TO_ID.items():
            f.write(f"  {v}: {k}\n")
            
    print(f"Saved dataset yaml to {yaml_path}")

if __name__ == "__main__":
    main()
