from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def repo_root(start: Optional[Path] = None) -> Path:
    """Best-effort find repo root (folder containing this file's ancestor)."""
    here = Path(__file__).resolve()
    return here.parents[2]


def dawn_raw_root(cli_value: Optional[str] = None) -> Path:
    """Resolve raw DAWN dataset root.

    Priority:
    1) CLI --raw_root
    2) env DAWN_RAW_ROOT
    3) default repo_root()/dataset/DAWN
    """
    if cli_value:
        return Path(cli_value).expanduser().resolve()

    env = os.environ.get("DAWN_RAW_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    return (repo_root() / "dataset" / "DAWN").resolve()


def dawn_processed_root() -> Path:
    return (repo_root() / "data" / "processed" / "dawn_yolo").resolve()
