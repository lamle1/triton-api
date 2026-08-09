"""
inference_utils.py — Shared parsing/validation for inference and uploads.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

# Triton model names: letters, digits, underscore, hyphen
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$")

IMGSZ_MIN = 32
IMGSZ_MAX = 4096
IMGSZ_STRIDE = 32
DEFAULT_IMGSZ = 640


class APIError(Exception):
    """Raised for client errors; map to HTTP status in main.py."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def parse_imgsz(
    imgsz: Optional[str | int],
    default: int = DEFAULT_IMGSZ,
) -> tuple[int, int]:
    """
    Parse client imgsz into (height, width) for letterbox preprocess.

    Accepts:
      None          → (default, default)
      640           → square
      "1280"        → square
      "1280,720"    → H, W
      "1280x720"    → H, W
    Values are clamped and rounded to IMGSZ_STRIDE.
    """
    if imgsz is None or imgsz == "":
        return (default, default)

    if isinstance(imgsz, int):
        parts = [imgsz]
    else:
        s = str(imgsz).strip().lower().replace("x", ",")
        parts = [p.strip() for p in s.split(",") if p.strip()]

    try:
        nums = [int(float(p)) for p in parts]
    except ValueError as exc:
        raise APIError(
            f"Invalid imgsz '{imgsz}': use an integer or 'height,width' (e.g. 640 or 1280,720)"
        ) from exc

    if len(nums) == 1:
        h = w = nums[0]
    elif len(nums) == 2:
        h, w = nums
    else:
        raise APIError("imgsz must be one number (square) or two: height,width")

    def _norm(v: int) -> int:
        v = max(IMGSZ_MIN, min(IMGSZ_MAX, v))
        return int(round(v / IMGSZ_STRIDE) * IMGSZ_STRIDE) or IMGSZ_STRIDE

    return (_norm(h), _norm(w))


def validate_model_name(name: str) -> str:
    """Return stripped name or raise APIError."""
    name = name.strip()
    if not name:
        raise APIError("Model name is empty")
    if not _MODEL_NAME_RE.match(name):
        raise APIError(
            "Invalid model name: use 1–63 chars, start with letter/digit, "
            "only letters, digits, underscore, hyphen"
        )
    return name


def parse_text_prompts(
    prompts: Optional[str],
    classes: Optional[str] = None,
) -> Optional[list[str]]:
    """
    YOLOE text prompts only.

    Prefer ``prompts``; ``classes`` is accepted as deprecated fallback for
    YOLOE-only calls so old clients keep working.
    """
    raw = prompts if prompts else classes
    if not raw or not str(raw).strip():
        return None
    return [p.strip() for p in str(raw).split(",") if p.strip()]


def default_model_name_from_filename(filename: Optional[str]) -> str:
    """
    Derive Triton model name from uploaded filename stem.
    e.g. 'best_helmet.pt' → 'best_helmet', 'yoloe-v8s-seg.pt' → 'yoloe-v8s-seg'
    """
    if not filename:
        raise APIError("Upload filename is required when 'name' is not set")
    stem = Path(filename).stem.strip()
    if not stem:
        raise APIError("Cannot derive model name from filename")
    # Collapse invalid chars to underscore
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    if not safe:
        raise APIError(f"Filename '{filename}' does not yield a valid model name")
    return validate_model_name(safe)
