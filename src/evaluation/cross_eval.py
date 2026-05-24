import os
import glob
import json
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
except ImportError:
    MeanAveragePrecision = None

from torchvision.utils import draw_bounding_boxes

# Import custom modules
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.filters.dark_channel_prior import dehaze
from src.utils.yolo26 import try_load_ultralytics_model

# Metrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Optional metrics
try:
    import lpips
    lpips_model = lpips.LPIPS(net='alex')
    if torch.cuda.is_available():
        lpips_model = lpips_model.cuda()
except:
    lpips_model = None

try:
    import pyiqa
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    niqe_metric = pyiqa.create_metric('niqe', as_loss=False).to(device)
    brisque_metric = pyiqa.create_metric('brisque', as_loss=False).to(device)
except:
    niqe_metric = None
    brisque_metric = None

logger = logging.getLogger(__name__)

def calc_lpips(img1_bgr, img2_bgr):
    if lpips_model is None: return 0.0
    img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)
    t1 = torch.from_numpy(img1_rgb).float().permute(2,0,1).unsqueeze(0)/127.5 - 1.0
    t2 = torch.from_numpy(img2_rgb).float().permute(2,0,1).unsqueeze(0)/127.5 - 1.0
    if torch.cuda.is_available():
        t1, t2 = t1.cuda(), t2.cuda()
    with torch.no_grad():
        dist = lpips_model(t1, t2)
    return dist.item()

def calc_no_ref(img_bgr):
    if niqe_metric is None: return 0.0, 0.0
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).float().permute(2,0,1).unsqueeze(0)/255.0
    t = t.to(device)
    with torch.no_grad():
        return niqe_metric(t).item(), brisque_metric(t).item()

def evaluate_reside(split_file, reside_root, patch_size, omega, t0, out_dir=None):
    with open(split_file, 'r') as f:
        test_pairs = json.load(f).get('test', [])
        
    if out_dir:
        os.makedirs(os.path.join(out_dir, "reside_dehazed"), exist_ok=True)
    
    psnrs, ssims, lpipss = [], [], []
    logger.info(f"Evaluating RESIDE on {len(test_pairs)} test pairs...")
    
    for pair in test_pairs:
        hazy_path = pair['hazy']
        clear_path = pair['clear']
        
        # Resolve relative paths
        if not os.path.isabs(hazy_path) and reside_root:
            hazy_path = os.path.join(reside_root, hazy_path)
        if not os.path.isabs(clear_path) and reside_root:
            clear_path = os.path.join(reside_root, clear_path)
            
        hazy = cv2.imread(hazy_path)
        clear = cv2.imread(clear_path)
        
        if hazy is None or clear is None: 
            logger.warning(f"Could not load image pair: {hazy_path} / {clear_path}")
            continue
            
        if hazy.shape != clear.shape:
            hazy = cv2.resize(hazy, (clear.shape[1], clear.shape[0]))
            
        dehazed = dehaze(hazy, patch_size=patch_size, omega=omega, t0=t0, refine=True)
        
        if out_dir:
            img_name = os.path.basename(pair['hazy'])
            cv2.imwrite(os.path.join(out_dir, "reside_dehazed", img_name), dehazed)
            
        psnrs.append(psnr(clear, dehazed))
        ssims.append(ssim(clear, dehazed, channel_axis=2))
        lpipss.append(calc_lpips(dehazed, clear))
        
    return {
        "PSNR": np.mean(psnrs) if psnrs else 0.0,
        "SSIM": np.mean(ssims) if ssims else 0.0,
        "LPIPS": np.mean(lpipss) if lpipss else 0.0
    }

def read_yolo_labels(label_path, img_w, img_h):
    boxes = []
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:5])
                    # unnormalize
                    x1 = (x_c - w/2) * img_w
                    y1 = (y_c - h/2) * img_h
                    x2 = (x_c + w/2) * img_w
                    y2 = (y_c + h/2) * img_h
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls_id)
    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

def evaluate_dawn(dawn_dir, patch_size, omega, t0, out_dir=None):
    test_img_dir = os.path.join(dawn_dir, "images", "test")
    test_lbl_dir = os.path.join(dawn_dir, "labels", "test")
    
    if out_dir:
        os.makedirs(os.path.join(out_dir, "dawn_detected"), exist_ok=True)
    
    image_paths = []
    fog_dir = os.path.join(test_img_dir, "Fog")
    if os.path.exists(fog_dir):
        image_paths.extend(glob.glob(os.path.join(fog_dir, "**", "*.jpg"), recursive=True))
        image_paths.extend(glob.glob(os.path.join(fog_dir, "**", "*.png"), recursive=True))
        
    logger.info(f"Evaluating DAWN (Fog only) on {len(image_paths)} test images...")
    
    # Load YOLO
    load_res = try_load_ultralytics_model("yolo26n.pt", allow_fallback=True)
    if not load_res.ok:
        logger.error("Could not load YOLO model for evaluation.")
        return {}
    yolo_model = load_res.model
    
    niqes, brisques = [], []
    
    # For torchmetrics MAP
    preds = []
    targets = []
    
    for impath in image_paths:
        hazy = cv2.imread(impath)
        if hazy is None: continue
        h, w = hazy.shape[:2]
        
        # Dehaze
        dehazed = dehaze(hazy, patch_size=patch_size, omega=omega, t0=t0, refine=True)
        
        # 1. No-ref metrics
        n, b = calc_no_ref(dehazed)
        niqes.append(n)
        brisques.append(b)
        
        # 2. YOLO inference
        if MeanAveragePrecision is not None:
            results = yolo_model(dehazed, verbose=False)[0]
            pred_boxes_raw = results.boxes.xyxy.cpu()
            pred_scores_raw = results.boxes.conf.cpu()
            pred_labels_raw = results.boxes.cls.cpu().to(torch.int64)
            
            # Map COCO to DAWN (0: vehicle, 1: person)
            mapped_boxes = []
            mapped_scores = []
            mapped_labels = []
            for box, score, lbl in zip(pred_boxes_raw, pred_scores_raw, pred_labels_raw):
                lbl = int(lbl)
                if lbl == 0: # COCO person
                    mapped_boxes.append(box)
                    mapped_scores.append(score)
                    mapped_labels.append(1) # DAWN person
                elif lbl in [2, 3, 5, 7]: # COCO vehicle (car, motorcycle, bus, truck)
                    mapped_boxes.append(box)
                    mapped_scores.append(score)
                    mapped_labels.append(0) # DAWN vehicle
                    
            if len(mapped_boxes) > 0:
                pred_boxes = torch.stack(mapped_boxes)
                pred_scores = torch.tensor(mapped_scores)
                pred_labels = torch.tensor(mapped_labels, dtype=torch.int64)
            else:
                pred_boxes = torch.empty((0,4))
                pred_scores = torch.empty((0,))
                pred_labels = torch.empty((0,), dtype=torch.int64)
            
            # Ground truth
            rel_path = os.path.relpath(impath, test_img_dir)
            lbl_path = os.path.join(test_lbl_dir, os.path.splitext(rel_path)[0] + ".txt")
            gt_boxes_np, gt_labels_np = read_yolo_labels(lbl_path, w, h)
            
            gt_boxes = torch.from_numpy(gt_boxes_np)
            gt_labels = torch.from_numpy(gt_labels_np)
            
            if len(gt_boxes) > 0:
                preds.append(dict(boxes=pred_boxes, scores=pred_scores, labels=pred_labels))
                targets.append(dict(boxes=gt_boxes, labels=gt_labels))
            else:
                preds.append(dict(boxes=pred_boxes, scores=pred_scores, labels=pred_labels))
                targets.append(dict(boxes=torch.empty((0,4)), labels=torch.empty((0,), dtype=torch.int64)))
                
            if out_dir:
                save_path = os.path.join(out_dir, "dawn_detected", rel_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # Convert BGR to RGB for torchvision
                img_rgb = cv2.cvtColor(dehazed, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).to(torch.uint8)
                
                # Draw Ground Truth Boxes
                if len(gt_boxes) > 0:
                    gt_str_labels = [f"GT:{'person' if l == 1 else 'vehicle'}" for l in gt_labels]
                    img_tensor = draw_bounding_boxes(img_tensor, gt_boxes, labels=gt_str_labels, colors="red", width=2)
                
                # Draw Predicted Boxes
                if len(pred_boxes) > 0:
                    pr_str_labels = [f"PR:{'person' if l == 1 else 'vehicle'} {s:.2f}" for l, s in zip(pred_labels, pred_scores)]
                    img_tensor = draw_bounding_boxes(img_tensor, pred_boxes, labels=pr_str_labels, colors="green", width=2)
                
                # Convert back to BGR and save
                drawn_rgb = img_tensor.permute(1, 2, 0).numpy()
                annotated_img = cv2.cvtColor(drawn_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, annotated_img)
        else:
            if out_dir:
                rel_path = os.path.relpath(impath, test_img_dir)
                save_path = os.path.join(out_dir, "dawn_detected", rel_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, dehazed)
    
    dawn_res = {
        "NIQE": np.mean(niqes) if niqes else 0.0,
        "BRISQUE": np.mean(brisques) if brisques else 0.0
    }
    
    if MeanAveragePrecision is not None and preds:
        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        metric.update(preds, targets)
        m = metric.compute()
        dawn_res["mAP@0.5:0.95"] = m["map"].item()
        dawn_res["mAP@0.5"] = m["map_50"].item()
        dawn_res["Recall@100"] = m["mar_100"].item()
        
    return dawn_res

if __name__ == "__main__":
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", type=int, default=15)
    parser.add_argument("--omega", type=float, default=0.95)
    parser.add_argument("--t0", type=float, default=0.1)
    parser.add_argument("--sots_split", default="configs/reside6k.json")
    parser.add_argument("--reside_root", default=r"C:\Users\manh hung\.cache\kagglehub\datasets\kmljts\reside-6k\versions\1\RESIDE-6K")
    parser.add_argument("--dawn_dir", default="data/processed/dawn_yolo")
    parser.add_argument("--method", default="DCP", help="Tên phương pháp đang dùng (ex: DCP)")
    parser.add_argument("--save_output", action="store_true", help="Bật cờ này để lưu ảnh output")
    parser.add_argument("--out_dir", default=None, help="Đường dẫn thư mục lưu kết quả. Nếu rỗng, tự động tạo tên theo method, config và thời gian")
    
    args = parser.parse_args()
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    if args.save_output:
        if not args.out_dir:
            run_name = f"{args.method}_p{args.patch_size}_o{args.omega}_t{args.t0}_{timestamp}"
            args.out_dir = os.path.join("results", run_name)
        os.makedirs(args.out_dir, exist_ok=True)

    # Setup Logging
    log_handlers = [logging.StreamHandler(sys.stdout)]
    if args.out_dir:
        log_file = os.path.join(args.out_dir, "run.log")
        log_handlers.append(logging.FileHandler(log_file, mode='w', encoding='utf-8'))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=log_handlers
    )

    if args.save_output:
        # Lưu config
        with open(os.path.join(args.out_dir, "config.json"), 'w') as f:
            json.dump({
                "method": args.method,
                "patch_size": args.patch_size,
                "omega": args.omega,
                "t0": args.t0,
                "timestamp": timestamp,
                "sots_split": args.sots_split,
                "reside_root": args.reside_root
            }, f, indent=4)
        logger.info(f"Outputs will be saved to: {args.out_dir}")
    
    logger.info("=== CROSS EVALUATION ===")
    logger.info(f"Config: patch={args.patch_size}, omega={args.omega}, t0={args.t0}")
    
    reside_metrics = evaluate_reside(args.sots_split, args.reside_root, args.patch_size, args.omega, args.t0, args.out_dir)
    logger.info("\n[RESIDE SOTS Test]")
    for k, v in reside_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
        
    dawn_metrics = evaluate_dawn(args.dawn_dir, args.patch_size, args.omega, args.t0, args.out_dir)
    logger.info("\n[DAWN YOLO Test]")
    for k, v in dawn_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
        
    if args.out_dir:
        # Lưu metrics log
        with open(os.path.join(args.out_dir, "metrics.json"), 'w') as f:
            json.dump({
                "RESIDE_SOTS": reside_metrics,
                "DAWN_YOLO": dawn_metrics
            }, f, indent=4)
            
        # Lưu vào file CSV tổng hợp
        csv_file = "results/summary_results.csv"
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, 'a', encoding='utf-8') as f:
            # Ghi header nếu file mới
            if not file_exists:
                f.write("Date,Method,PatchSize,Omega,t0,PSNR,SSIM,LPIPS,NIQE,BRISQUE,mAP50-95,mAP50,Recall,OutDir\n")
            
            # Ghi data
            psnr_val = reside_metrics.get("PSNR", 0)
            ssim_val = reside_metrics.get("SSIM", 0)
            lpips_val = reside_metrics.get("LPIPS", 0)
            niqe_val = dawn_metrics.get("NIQE", 0)
            brisque_val = dawn_metrics.get("BRISQUE", 0)
            map_all = dawn_metrics.get("mAP@0.5:0.95", 0)
            map_50 = dawn_metrics.get("mAP@0.5", 0)
            recall = dawn_metrics.get("Recall@100", 0)
            
            f.write(f"{timestamp},{args.method},{args.patch_size},{args.omega},{args.t0},"
                    f"{psnr_val:.4f},{ssim_val:.4f},{lpips_val:.4f},"
                    f"{niqe_val:.4f},{brisque_val:.4f},"
                    f"{map_all:.4f},{map_50:.4f},{recall:.4f},{args.out_dir}\n")
        logger.info(f"\n=> Đã ghi log tổng hợp vào: {csv_file}")
