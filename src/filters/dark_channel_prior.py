"""Dark Channel Prior (DCP) single-image dehazing.

Reference:
  K. He, J. Sun, X. Tang, "Single Image Haze Removal Using Dark Channel Prior", CVPR 2009.

Notes:
- Purely classical computer vision (no deep learning).
- Accepts RGB or BGR arrays; channel order does not affect the core DCP math.
- Uses a guided filter (He et al.)-style refinement for the transmission map.

CLI:
  python -m src.filters.dark_channel_prior --input path/to/img.jpg --output out.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

# Allow running as module or script from repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np


def _to_float01(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape={image.shape}")

    if image.dtype == np.uint8:
        out = image.astype(np.float32) / 255.0
    else:
        out = image.astype(np.float32)
        # If data looks like uint8-range but stored as float, normalize.
        if out.max(initial=0.0) > 1.5:
            out = out / 255.0

    return np.clip(out, 0.0, 1.0)


def _to_uint8(image01: np.ndarray) -> np.ndarray:
    image01 = np.clip(image01, 0.0, 1.0)
    return (image01 * 255.0 + 0.5).astype(np.uint8)


def _odd_ksize(patch_size: int) -> int:
    p = int(patch_size)
    if p <= 1:
        return 1
    if p % 2 == 0:
        p += 1
    return p


def dark_channel(image: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """Compute the dark channel.

    Args:
        image: HxWx3 (uint8 or float). RGB or BGR.
        patch_size: local min filter size.

    Returns:
        dark: HxW float32 in [0,1].
    """

    img = _to_float01(image)
    min_rgb = np.min(img, axis=2)

    k = _odd_ksize(patch_size)
    if k <= 1:
        return min_rgb.astype(np.float32)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    dark = cv2.erode(min_rgb, kernel)
    return dark.astype(np.float32)


def estimate_atmospheric_light(
    image: np.ndarray,
    dark_channel: np.ndarray,
    top_percent: float = 0.001,
) -> np.ndarray:
    """Estimate the atmospheric light A from the brightest dark-channel pixels.

    Standard approach: take the top p% pixels in the dark channel (largest values),
    then pick the pixel with the highest intensity in the original image.

    Returns:
        A: shape (3,) float32 in [0,1]
    """

    img = _to_float01(image)

    if dark_channel.ndim != 2:
        raise ValueError("dark_channel must be HxW")

    h, w = dark_channel.shape
    n = h * w
    p = float(top_percent)
    if not (0.0 < p <= 1.0):
        raise ValueError("top_percent must be in (0, 1]")

    k = max(1, int(round(n * p)))

    dark_flat = dark_channel.reshape(-1)
    idx = np.argpartition(dark_flat, -k)[-k:]

    img_flat = img.reshape(-1, 3)
    candidates = img_flat[idx]

    # Highest brightness (sum over channels) among candidates.
    brightness = np.sum(candidates, axis=1)
    best = int(idx[int(np.argmax(brightness))])
    A = img_flat[best]

    return np.clip(A.astype(np.float32), 0.0, 1.0)


def estimate_transmission(
    image: np.ndarray,
    atmospheric_light: np.ndarray,
    omega: float = 0.95,
    patch_size: int = 15,
) -> np.ndarray:
    """Estimate initial transmission map t(x)."""

    img = _to_float01(image)
    A = np.asarray(atmospheric_light, dtype=np.float32).reshape(1, 1, 3)
    A = np.clip(A, 1e-6, 1.0)  # avoid division by zero

    normed = img / A
    dc = dark_channel(normed, patch_size=patch_size)

    t = 1.0 - float(omega) * dc
    return np.clip(t.astype(np.float32), 0.0, 1.0)


def guided_filter(
    guidance: np.ndarray,
    src: np.ndarray,
    radius: int = 40,
    eps: float = 1e-3,
) -> np.ndarray:
    """Edge-preserving guided filter (grayscale guidance).

    Args:
        guidance: HxW float32 in [0,1] (guidance image, e.g. grayscale).
        src: HxW float32 (filter input, e.g. transmission).
        radius: window radius r.
        eps: regularization.

    Returns:
        filtered: HxW float32.
    """

    I = guidance.astype(np.float32)
    p = src.astype(np.float32)

    r = int(max(1, radius))
    win = (2 * r + 1, 2 * r + 1)

    mean_I = cv2.boxFilter(I, ddepth=-1, ksize=win, normalize=True)
    mean_p = cv2.boxFilter(p, ddepth=-1, ksize=win, normalize=True)
    mean_Ip = cv2.boxFilter(I * p, ddepth=-1, ksize=win, normalize=True)

    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, ddepth=-1, ksize=win, normalize=True)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + float(eps))
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, ddepth=-1, ksize=win, normalize=True)
    mean_b = cv2.boxFilter(b, ddepth=-1, ksize=win, normalize=True)

    q = mean_a * I + mean_b
    return q.astype(np.float32)


def recover_radiance(
    image: np.ndarray,
    transmission: np.ndarray,
    atmospheric_light: np.ndarray,
    t0: float = 0.1,
) -> np.ndarray:
    """Recover the scene radiance (dehazed image) J."""

    img = _to_float01(image)
    A = np.asarray(atmospheric_light, dtype=np.float32).reshape(1, 1, 3)
    A = np.clip(A, 0.0, 1.0)

    t = transmission.astype(np.float32)
    if t.ndim != 2:
        raise ValueError("transmission must be HxW")

    t = np.maximum(t, float(t0))
    t3 = t[:, :, None]

    J = (img - A) / t3 + A
    return np.clip(J.astype(np.float32), 0.0, 1.0)


def dehaze(
    image: np.ndarray,
    patch_size: int = 15,
    omega: float = 0.95,
    t0: float = 0.1,
    *,
    top_percent: float = 0.001,
    refine: bool = True,
    guided_radius: Optional[int] = None,
    guided_eps: float = 1e-3,
) -> np.ndarray:
    """Full DCP dehazing pipeline.

    Returns:
        Enhanced uint8 image, same shape HxWx3.
    """

    img01 = _to_float01(image)

    dc = dark_channel(img01, patch_size=patch_size)
    A = estimate_atmospheric_light(img01, dc, top_percent=top_percent)
    t_est = estimate_transmission(img01, A, omega=omega, patch_size=patch_size)

    if refine:
        gray = img01.mean(axis=2).astype(np.float32)
        r = guided_radius
        if r is None:
            # Typical guided filter radius is larger than the dark-channel patch.
            r = max(8, int(round(_odd_ksize(patch_size) * 2.0)))
        t_ref = guided_filter(gray, t_est, radius=r, eps=guided_eps)
        t_ref = np.clip(t_ref, 0.0, 1.0)
    else:
        t_ref = t_est

    J01 = recover_radiance(img01, t_ref, A, t0=t0)
    return _to_uint8(J01)


def _concat_before_after(before_bgr: np.ndarray, after_bgr: np.ndarray) -> np.ndarray:
    if before_bgr.shape != after_bgr.shape:
        raise ValueError("before/after shapes must match")
    return np.concatenate([before_bgr, after_bgr], axis=1)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dark Channel Prior (DCP) dehazing")
    p.add_argument("--input", required=False, help="Input image path")
    p.add_argument("--output", required=False, help="Output image path")

    p.add_argument("--patch_size", type=int, default=15)
    p.add_argument("--omega", type=float, default=0.95)
    p.add_argument("--t0", type=float, default=0.1)
    p.add_argument("--top_percent", type=float, default=0.001)

    p.add_argument("--no_refine", action="store_true", help="Disable guided-filter refinement")
    p.add_argument("--guided_radius", type=int, default=None)
    p.add_argument("--guided_eps", type=float, default=1e-3)

    p.add_argument(
        "--viz",
        default=None,
        help="Optional path to save a side-by-side before/after visualization",
    )

    p.add_argument(
        "--test",
        action="store_true",
        help="Run a small synthetic haze sanity-check instead of processing an input image",
    )

    return p.parse_args()


def test_synthetic_haze(seed: int = 0) -> Tuple[float, float]:
    """Quick sanity test: synthetic haze should be partially removed.

    Returns:
        (mse_hazy_vs_clean, mse_dehazed_vs_clean)
    """

    rng = np.random.default_rng(seed)
    h, w = 240, 320

    clean = rng.uniform(0.0, 1.0, size=(h, w, 3)).astype(np.float32)
    # Smooth the clean image a bit to resemble natural content.
    clean = cv2.GaussianBlur(clean, (0, 0), 1.2)

    # Depth-like ramp: left is near (high t), right is far (low t).
    depth = np.tile(np.linspace(0.0, 1.0, w, dtype=np.float32), (h, 1))
    t = (0.15 + 0.85 * (1.0 - depth)).astype(np.float32)  # in [0.15,1]

    A = np.array([0.85, 0.85, 0.85], dtype=np.float32).reshape(1, 1, 3)
    hazy = clean * t[:, :, None] + A * (1.0 - t[:, :, None])

    dehazed_u8 = dehaze((hazy * 255.0).astype(np.uint8))
    dehazed = _to_float01(dehazed_u8)

    mse_hazy = float(np.mean((hazy - clean) ** 2))
    mse_dehazed = float(np.mean((dehazed - clean) ** 2))
    return mse_hazy, mse_dehazed


def main() -> None:
    args = _parse_args()

    if args.test:
        mse_hazy, mse_dehazed = test_synthetic_haze()
        print(f"Synthetic haze MSE (hazy vs clean):    {mse_hazy:.6f}")
        print(f"Synthetic haze MSE (dehazed vs clean): {mse_dehazed:.6f}")
        if mse_dehazed > mse_hazy:
            print("Warning: dehazing did not improve MSE on synthetic test (still may look better visually).")
        return

    if not args.input or not args.output:
        raise SystemExit("--input and --output are required unless --test is used")

    in_path = Path(args.input)
    out_path = Path(args.output)

    img_bgr = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise SystemExit(f"Failed to read image: {in_path}")

    out_bgr = dehaze(
        img_bgr,
        patch_size=args.patch_size,
        omega=args.omega,
        t0=args.t0,
        top_percent=args.top_percent,
        refine=not args.no_refine,
        guided_radius=args.guided_radius,
        guided_eps=args.guided_eps,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), out_bgr)
    if not ok:
        raise SystemExit(f"Failed to write: {out_path}")

    if args.viz:
        viz_path = Path(args.viz)
        viz_path.parent.mkdir(parents=True, exist_ok=True)
        viz = _concat_before_after(img_bgr, out_bgr)
        ok2 = cv2.imwrite(str(viz_path), viz)
        if not ok2:
            raise SystemExit(f"Failed to write viz: {viz_path}")


if __name__ == "__main__":
    main()
