from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_data_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid dataset yaml: {p}")
    return data


def class_names_from_data(data: Dict[str, Any]) -> List[str]:
    names = data.get("names")
    if isinstance(names, dict):
        # keys might be strings
        items = [(int(k), str(v)) for k, v in names.items()]
        return [v for _, v in sorted(items, key=lambda x: x[0])]
    if isinstance(names, list):
        return [str(x) for x in names]
    raise ValueError("Dataset yaml missing valid 'names' field")
