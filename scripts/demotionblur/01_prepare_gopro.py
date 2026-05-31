import os
import random
import csv
import argparse

def prepare_gopro(dataset_dir, output_csv, sample_size=1000):
    input_dir = os.path.join(dataset_dir, 'input')
    sharp_dir = os.path.join(dataset_dir, 'restored')  # assuming restored acts as sharp GT here
    
    # Alternatively if 'target' exists, use it
    if os.path.exists(os.path.join(dataset_dir, 'target')):
        sharp_dir = os.path.join(dataset_dir, 'target')
    elif os.path.exists(os.path.join(dataset_dir, 'sharp')):
        sharp_dir = os.path.join(dataset_dir, 'sharp')

    if not os.path.exists(input_dir) or not os.path.exists(sharp_dir):
        print(f"Error: Missing input or sharp directory in {dataset_dir}")
        return

    # Gather pairs
    input_files = set(f for f in os.listdir(input_dir) if f.endswith('.png'))
    sharp_files = set(f for f in os.listdir(sharp_dir) if f.endswith('.png'))
    
    common_files = list(input_files.intersection(sharp_files))
    missing_sharp = input_files - sharp_files
    missing_blur = sharp_files - input_files
    
    if missing_sharp:
        print(f"Warning: {len(missing_sharp)} blurry images have no matching sharp images.")
    if missing_blur:
        print(f"Warning: {len(missing_blur)} sharp images have no matching blurry images.")
        
    print(f"Found {len(common_files)} matching pairs.")
    
    # Shuffle and subsample
    random.seed(42)
    random.shuffle(common_files)
    
    if len(common_files) > sample_size:
        common_files = common_files[:sample_size]
        print(f"Subsampled to {sample_size} pairs.")
        
    # Split 20% val, 80% test
    num_val = int(len(common_files) * 0.2)
    val_files = common_files[:num_val]
    test_files = common_files[num_val:]
    
    print(f"Validation set: {len(val_files)} pairs.")
    print(f"Test set: {len(test_files)} pairs.")
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'image_id', 'blur_path', 'sharp_path'])
        
        for img in val_files:
            blur_path = os.path.join(input_dir, img)
            sharp_path = os.path.join(sharp_dir, img)
            writer.writerow(['val', img, blur_path, sharp_path])
            
        for img in test_files:
            blur_path = os.path.join(input_dir, img)
            sharp_path = os.path.join(sharp_dir, img)
            writer.writerow(['test', img, blur_path, sharp_path])

    print(f"Saved pair index to {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default=r'A:\HUST_on_GitHub\ProjectCV\dataset\GoPro')
    parser.add_argument('--output_csv', type=str, default=r'A:\HUST_on_GitHub\ProjectCV\outputs\motion_deblur\gopro_pairs.csv')
    parser.add_argument('--sample_size', type=int, default=1000)
    args = parser.parse_args()
    
    prepare_gopro(args.dataset_dir, args.output_csv, args.sample_size)
