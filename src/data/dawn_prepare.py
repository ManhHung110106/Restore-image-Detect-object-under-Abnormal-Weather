"""DAWN dataset preparation utilities.

Goal: convert DAWN PASCAL VOC annotations into an Ultralytics-compatible YOLO dataset layout:

  data/processed/dawn_yolo/
    images/{train,val,test}/{Fog,Rain,Snow,Sand}/*.jpg
    labels/{train,val,test}/{Fog,Rain,Snow,Sand}/*.txt

We do NOT apply any preprocessing filters here (baseline).

Design notes:
- We use the VOC XML files as source of truth for class names.
- For reproducibility, we stratify splits by weather condition with a fixed seed.
- We avoid silently inventing class mappings: the class list is explicit and fixed.

This module is used by:
- src/detection/train_yolo.py
- src/detection/evaluate_yolo.py
"""

from __future__ import annotations

import os
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


WEATHER_FOLDERS: Tuple[str, ...] = ("Fog", "Rain", "Snow", "Sand")

# Fixed label space for the baseline experiment (per user requirement).
# Keep stable ordering across runs.
BASELINE_CLASSES: Tuple[str, ...] = ("vehicle", "person")

# Common DAWN/VOC spelling variants.
_CLASS_SYNONYMS: Dict[str, str] = {
    "people": "person",
    "pedestrian": "person",
    "motorcycle": "motorbike",
    "motorbike": "motorbike",
    "bike": "motorbike",  # conservative; may be wrong for bicycle-only datasets
}


def _map_to_target_label(voc_label: str, *, classes: Sequence[str]) -> Optional[str]:
    """Map a raw VOC label into the caller's target label space.

    If the target label space includes 'vehicle', we map common transport labels
    (car/bus/train/motorbike/...) into 'vehicle'. Person stays as 'person'.

    Otherwise, we only keep labels that exactly match the provided class list.
    """

    cls_set = {c.lower() for c in classes}
    raw = normalize_class_name(voc_label)

    if "vehicle" in cls_set and raw in {"car", "bus", "train", "motorbike", "motorcycle", "truck", "van"}:
        return "vehicle"
    if "person" in cls_set and raw in {"person", "people", "pedestrian"}:
        return "person"

    if raw in cls_set:
        return raw

    return None


@dataclass(frozen=True)
class DawnItem:
    weather: str
    image_path: Path
    xml_path: Path


def normalize_class_name(name: str) -> str:
    key = name.strip().lower()
    return _CLASS_SYNONYMS.get(key, key)


def _iter_voc_items(raw_root: Path, weathers: Sequence[str]) -> List[DawnItem]:
    items: List[DawnItem] = []
    for weather in weathers:
        weather_dir = raw_root / weather
        if not weather_dir.exists():
            continue

        voc_dir = None
        # DAWN layout observed in workspace:
        #   dataset/DAWN/Fog/Fog_PASCAL_VOC/*.xml
        #   dataset/DAWN/Rain/Rain_PASCAL_VOC/*.xml
        #   dataset/DAWN/Snow/Snow_PASCAL_VOC/*.xml
        #   dataset/DAWN/Sand/Sand_PASCAL_VOC/*.xml
        for candidate in weather_dir.glob("*_PASCAL_VOC"):
            if candidate.is_dir():
                voc_dir = candidate
                break

        if voc_dir is None:
            # fallback: any xml under this weather folder
            xmls = list(weather_dir.rglob("*.xml"))
        else:
            xmls = list(voc_dir.glob("*.xml"))

        for xml_path in sorted(xmls):
            stem = xml_path.stem
            img_candidates = [
                weather_dir / f"{stem}.jpg",
                weather_dir / f"{stem}.png",
                weather_dir / f"{stem}.jpeg",
            ]
            image_path = next((p for p in img_candidates if p.exists()), None)
            if image_path is None:
                # search by name anywhere under weather_dir
                by_name = list(weather_dir.rglob(f"{stem}.jpg"))
                if not by_name:
                    by_name = list(weather_dir.rglob(f"{stem}.png"))
                if not by_name:
                    by_name = list(weather_dir.rglob(f"{stem}.jpeg"))
                image_path = by_name[0] if by_name else None

            if image_path is None:
                continue

            items.append(DawnItem(weather=weather, image_path=image_path, xml_path=xml_path))

    return items


def _parse_voc_boxes(xml_path: Path) -> Tuple[int, int, List[Tuple[str, Tuple[float, float, float, float]]]]:
    """Return (width, height, objects) where objects = [(class_name, (xmin,ymin,xmax,ymax)), ...]."""
    root = ET.fromstring(xml_path.read_text(encoding="utf-8", errors="ignore"))

    size = root.find("size")
    if size is None:
        raise ValueError(f"VOC XML missing <size>: {xml_path}")

    width = int(size.findtext("width", default="0"))
    height = int(size.findtext("height", default="0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in XML: {xml_path}")

    objects: List[Tuple[str, Tuple[float, float, float, float]]] = []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="").strip()
        if not name:
            continue
        cls = normalize_class_name(name)

        bnd = obj.find("bndbox")
        if bnd is None:
            continue

        xmin = float(bnd.findtext("xmin", default="0"))
        ymin = float(bnd.findtext("ymin", default="0"))
        xmax = float(bnd.findtext("xmax", default="0"))
        ymax = float(bnd.findtext("ymax", default="0"))

        # Clamp to image bounds (VOC coords are 1-based sometimes; DAWN looks 0-based ints but be safe)
        xmin = max(0.0, min(xmin, width - 1.0))
        ymin = max(0.0, min(ymin, height - 1.0))
        xmax = max(0.0, min(xmax, width - 1.0))
        ymax = max(0.0, min(ymax, height - 1.0))

        if xmax <= xmin or ymax <= ymin:
            continue

        objects.append((cls, (xmin, ymin, xmax, ymax)))

    return width, height, objects


def _xyxy_to_yolo(xmin: float, ymin: float, xmax: float, ymax: float, *, w: int, h: int) -> Tuple[float, float, float, float]:
    bw = (xmax - xmin) / float(w)
    bh = (ymax - ymin) / float(h)
    xc = (xmin + xmax) / 2.0 / float(w)
    yc = (ymin + ymax) / 2.0 / float(h)
    return xc, yc, bw, bh


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    try:
        os.link(src, dst)  # hardlink if possible (fast, no duplication)
        return
    except Exception:
        pass

    try:
        # symlink as second choice (may require admin on Windows)
        os.symlink(src, dst)
        return
    except Exception:
        pass

    shutil.copy2(src, dst)


def make_split(items: Sequence[DawnItem], *, seed: int, ratios: Tuple[float, float, float]) -> Dict[str, List[DawnItem]]:
    """Return dict(split -> items) with stratification by weather."""
    if len(ratios) != 3:
        raise ValueError("ratios must be (train,val,test)")
    r_train, r_val, r_test = ratios
    if abs((r_train + r_val + r_test) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")

    by_weather: Dict[str, List[DawnItem]] = {}
    for it in items:
        by_weather.setdefault(it.weather, []).append(it)

    rng = random.Random(seed)
    out: Dict[str, List[DawnItem]] = {"train": [], "val": [], "test": []}

    for weather, group in by_weather.items():
        group = list(group)
        rng.shuffle(group)
        n = len(group)
        n_train = int(round(n * r_train))
        n_val = int(round(n * r_val))
        # ensure total = n
        n_train = min(max(n_train, 0), n)
        n_val = min(max(n_val, 0), n - n_train)
        n_test = n - n_train - n_val

        out["train"].extend(group[:n_train])
        out["val"].extend(group[n_train : n_train + n_val])
        out["test"].extend(group[n_train + n_val :])

    return out


def prepare_dawn_yolo(
    *,
    raw_root: Path,
    out_root: Path,
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    weathers: Sequence[str] = WEATHER_FOLDERS,
    classes: Sequence[str] = BASELINE_CLASSES,
) -> Path:
    """Create/refresh the YOLO dataset. Returns out_root."""

    raw_root = raw_root.resolve()
    out_root = out_root.resolve()

    items = _iter_voc_items(raw_root, weathers)
    if not items:
        raise FileNotFoundError(
            f"No DAWN VOC items found under {raw_root}. Expected DAWN layout like: {raw_root}/Fog/Fog_PASCAL_VOC/*.xml"
        )

    splits = make_split(items, seed=seed, ratios=ratios)

    # Refresh output to avoid stale splits/labels when seed/ratios/classes change.
    for sub in ("images", "labels"):
        d = out_root / sub
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

    class_to_id = {c.lower(): i for i, c in enumerate(classes)}

    for split_name, split_items in splits.items():
        for it in split_items:
            # Destination paths preserve weather for per-weather metrics.
            dst_img = out_root / "images" / split_name / it.weather / it.image_path.name
            dst_lbl = out_root / "labels" / split_name / it.weather / (it.image_path.stem + ".txt")

            _safe_link_or_copy(it.image_path, dst_img)

            w, h, objs = _parse_voc_boxes(it.xml_path)
            lines: List[str] = []
            for voc_cls, (xmin, ymin, xmax, ymax) in objs:
                mapped = _map_to_target_label(voc_cls, classes=classes)
                if mapped is None:
                    continue
                cls_key = mapped.lower()
                if cls_key not in class_to_id:
                    continue
                xc, yc, bw, bh = _xyxy_to_yolo(xmin, ymin, xmax, ymax, w=w, h=h)
                # YOLO expects normalized in [0,1]
                xc = min(max(xc, 0.0), 1.0)
                yc = min(max(yc, 0.0), 1.0)
                bw = min(max(bw, 0.0), 1.0)
                bh = min(max(bh, 0.0), 1.0)
                lines.append(f"{class_to_id[cls_key]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    # Metadata for reproducibility/debugging.
    meta = {
        "raw_root": str(raw_root),
        "out_root": str(out_root),
        "seed": int(seed),
        "ratios": list(ratios),
        "weathers": list(weathers),
        "classes": [str(c) for c in classes],
    }
    (out_root / "_meta.json").write_text(
        __import__("json").dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Write a minimal data.yaml into out_root so users can also point Ultralytics directly.
    data_yaml = out_root / "data.yaml"
    names_block = "\n".join([f"  {i}: {c}" for i, c in enumerate(classes)])
    data_yaml.write_text(
        "\n".join(
            [
                "# Auto-generated by src/data/dawn_prepare.py",
                f"path: {out_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                names_block,
                "",
            ]
        ),
        encoding="utf-8",
    )

    return out_root
