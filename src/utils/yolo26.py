"""Ultralytics YOLO26 model loading helpers.

Requirement behavior:
- Try to load YOLO26 weights (e.g., yolo26n.pt).
- If unavailable, print ultralytics version + clear guidance to update.
- Only allow fallback to closest available Ultralytics YOLO model if --allow_fallback is set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelLoadResult:
    ok: bool
    model: Optional[object]
    message: str


def ultralytics_version() -> str:
    try:
        import ultralytics

        return getattr(ultralytics, "__version__", "unknown")
    except Exception:
        return "not-installed"


def try_load_ultralytics_model(model_spec: str, *, allow_fallback: bool = False, fallback_model: str = "yolo26n.pt") -> ModelLoadResult:
    """Try to load model_spec with ultralytics.YOLO.

    Returns a ModelLoadResult. Never raises; caller decides to exit.
    """

    try:
        from ultralytics import YOLO

        model = YOLO(model_spec)
        return ModelLoadResult(ok=True, model=model, message=f"Loaded model: {model_spec}")
    except Exception as e:
        ver = ultralytics_version()
        msg = (
            "Failed to load YOLO26 model weights with Ultralytics YOLO.\n"
            f"- model spec: {model_spec}\n"
            f"- ultralytics version: {ver}\n"
            f"- error: {type(e).__name__}: {e}\n\n"
            "Action: please update Ultralytics and retry, e.g.\n"
            "  pip install -U ultralytics\n"
        )

        if not allow_fallback:
            msg += "\nFallback is disabled. Re-run with --allow_fallback to use a closest available YOLO model.\n"
            return ModelLoadResult(ok=False, model=None, message=msg)

        # fallback
        try:
            from ultralytics import YOLO

            fb = YOLO(fallback_model)
            msg += f"\nFallback enabled: loaded {fallback_model} instead.\n"
            return ModelLoadResult(ok=True, model=fb, message=msg)
        except Exception as e2:
            msg += f"\nFallback also failed: {type(e2).__name__}: {e2}\n"
            return ModelLoadResult(ok=False, model=None, message=msg)
