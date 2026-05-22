"""Batch preprocessing utilities for classical filters.

Currently supports:
- Dark Channel Prior (DCP) dehazing for foggy images.

Example:
  python -m src.filters.batch_preprocess \
    --filter dcp \
    --input_dir dataset/DAWN/Fog \
    --output_dir data/processed/dawn_dehazed/Fog \
    --config configs/dcp.yaml \
    --save_viz

This module is intentionally simple and does not alter labels/annotations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from src.filters.dark_channel_prior import dehaze


def _iter_images(root: Path, *, recursive: bool, exts: Tuple[str, ...]) -> Iterable[Path]:
    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                yield p
    else:
        for p in root.glob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                yield p


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config (expected mapping): {p}")
    return data


def _concat_before_after(before_bgr: np.ndarray, after_bgr: np.ndarray) -> np.ndarray:
    if before_bgr.shape != after_bgr.shape:
        raise ValueError("before/after shapes must match")
    return np.concatenate([before_bgr, after_bgr], axis=1)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch preprocess images (classical filters)")

    p.add_argument("--filter", default="dcp", choices=["dcp"], help="Which filter to apply")

    p.add_argument("--input_dir", required=True, help="Input folder containing images")
    p.add_argument("--output_dir", required=True, help="Output folder to write processed images")

    p.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    p.add_argument("--exts", default=".jpg,.jpeg,.png", help="Comma-separated extensions")

    p.add_argument("--config", default=None, help="Optional YAML config for filter parameters")

    p.add_argument("--save_viz", action="store_true", help="Save side-by-side before/after images")
    p.add_argument(
        "--viz_dir",
        default=None,
        help="Optional explicit directory for visualizations (default: output_dir/_viz)",
    )

    p.add_argument("--max_images", type=int, default=0, help="Limit images processed (0 = all)")

    return p.parse_args()


def run_batch(
    *,
    input_dir: Path,
    output_dir: Path,
    config: Optional[Dict[str, Any]] = None,
    recursive: bool = False,
    exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    save_viz: bool = False,
    viz_dir: Optional[Path] = None,
    max_images: int = 0,
) -> int:
    """Run the configured filter over a directory tree.

    Returns:
        count of processed images.
    """

    cfg = dict(config or {})
    # DCP parameter defaults are in the dehaze() function.
    dcp_cfg = cfg.get("dcp", cfg)  # allow either flat or nested under 'dcp'
    if not isinstance(dcp_cfg, dict):
        raise ValueError("config must be a mapping (or contain 'dcp' mapping)")

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_viz:
        viz_dir = (viz_dir or (output_dir / "_viz")).resolve()
        viz_dir.mkdir(parents=True, exist_ok=True)

    paths = list(_iter_images(input_dir, recursive=recursive, exts=exts))
    paths.sort()
    if max_images and max_images > 0:
        paths = paths[: int(max_images)]

    processed = 0
    for p in tqdm(paths, desc="preprocess", unit="img"):
        rel = p.relative_to(input_dir)
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        out_bgr = dehaze(
            img_bgr,
            patch_size=int(dcp_cfg.get("patch_size", 15)),
            omega=float(dcp_cfg.get("omega", 0.95)),
            t0=float(dcp_cfg.get("t0", 0.1)),
            top_percent=float(dcp_cfg.get("top_percent", 0.001)),
            refine=bool(dcp_cfg.get("refine", True)),
            guided_radius=dcp_cfg.get("guided_radius", None),
            guided_eps=float(dcp_cfg.get("guided_eps", 1e-3)),
        )

        ok = cv2.imwrite(str(out_path), out_bgr)
        if not ok:
            continue

        if save_viz and viz_dir is not None:
            viz_path = viz_dir / rel
            viz_path.parent.mkdir(parents=True, exist_ok=True)
            viz = _concat_before_after(img_bgr, out_bgr)
            cv2.imwrite(str(viz_path), viz)

        processed += 1

    return processed


def main() -> None:
    args = _parse_args()

    cfg = _load_yaml(args.config)
    exts = tuple(e.strip().lower() for e in str(args.exts).split(",") if e.strip())

    processed = run_batch(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        config=cfg,
        recursive=bool(args.recursive),
        exts=exts,
        save_viz=bool(args.save_viz),
        viz_dir=Path(args.viz_dir) if args.viz_dir else None,
        max_images=int(args.max_images),
    )

    print(f"Processed {processed} images")


if __name__ == "__main__":
    main()
