"""
Restoration evaluation utilities for comparing dehazing configs
"""
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def load_reside_samples(reside_root: str) -> List[Tuple[str, str]]:
    """
    Load RESIDE-6K dataset samples
    
    Args:
        reside_root: Root directory of RESIDE-6K dataset
    
    Returns:
        List of tuples (hazy_path, gt_path)
    """
    samples = [
        (os.path.join(reside_root, "train/hazy/1.jpg"), os.path.join(reside_root, "train/GT/1.jpg")),
        (os.path.join(reside_root, "train/hazy/10.jpg"), os.path.join(reside_root, "train/GT/10.jpg"))
    ]
    return samples


def compute_restoration_metrics_reside(gt, restored):
    """
    Compute PSNR and SSIM metrics for RESIDE-6K evaluation
    
    Args:
        gt: Ground truth image
        restored: Restored/dehazed image
    
    Returns:
        Dict with psnr and ssim values
    """
    p = psnr(gt, restored)
    s = ssim(gt, restored, channel_axis=2)
    return {"psnr": p, "ssim": s}


def compute_restoration_metrics_noreference(img_rgb, niqe_metric, brisque_metric, device):
    """
    Compute NIQE and BRISQUE metrics for DAWN evaluation (no reference)
    
    Args:
        img_rgb: Input image in RGB format
        niqe_metric: NIQE metric object
        brisque_metric: BRISQUE metric object
        device: torch device
    
    Returns:
        Dict with niqe and brisque values
    """
    img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        n = niqe_metric(img_tensor).item()
        b = brisque_metric(img_tensor).item()
    
    return {"niqe": n, "brisque": b}
