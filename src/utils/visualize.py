from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def draw_boxes_bgr(
    img_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    labels: np.ndarray,
    scores: Optional[np.ndarray],
    class_names: List[str],
    *,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    out = img_bgr.copy()
    thickness_i = max(1, int(thickness))
    for i in range(int(boxes_xyxy.shape[0])):
        x1, y1, x2, y2 = [int(v) for v in boxes_xyxy[i].tolist()]
        cls = int(labels[i])
        name = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        conf = float(scores[i]) if scores is not None else None
        txt = f"{name}" if conf is None else f"{name} {conf:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness_i)
        cv2.putText(
            out,
            txt,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            thickness_i,
        )
    return out


def save_image(path: Path, img_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")
