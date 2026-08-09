"""Ensure YOLOE MobileCLIP TorchScript asset is present and loadable."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

MOBILECLIP_ASSET = "mobileclip_blt.ts"
# Ultralytics resolves attempt_download_asset() relative to CWD (/app) first.
APP_ASSET_PATH = Path(os.getenv("MOBILECLIP_ASSET_PATH", "/app/mobileclip_blt.ts"))
# Persistent copy on the mounted weights volume (survives container recreate).
WEIGHTS_ASSET_PATH = Path(os.getenv("MOBILECLIP_WEIGHTS_PATH", "/weights/mobileclip_blt.ts"))
MIN_BYTES = 500_000_000  # full release is ~572 MB; partial downloads are much smaller


def _is_loadable(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < MIN_BYTES:
        return False
    try:
        torch.jit.load(str(path), map_location="cpu")
        return True
    except Exception:
        return False


def ensure_mobileclip_asset() -> Path:
    """
    Return a valid mobileclip_blt.ts path for Ultralytics YOLOE text encoding.

    Prefers the weights-volume copy, falls back to /app, and re-downloads if corrupt.
    """
    candidates = [WEIGHTS_ASSET_PATH, APP_ASSET_PATH]
    for path in candidates:
        if _is_loadable(path):
            logger.info("MobileCLIP asset OK: %s (%d MB)", path, path.stat().st_size // (1024 * 1024))
            if not _is_loadable(APP_ASSET_PATH):
                _link_app_asset(path)
            return path

    for path in candidates:
        if path.exists():
            logger.warning(
                "Removing corrupt/incomplete MobileCLIP asset %s (%d bytes)",
                path,
                path.stat().st_size,
            )
            path.unlink()

    from ultralytics.utils.downloads import attempt_download_asset

    dest = WEIGHTS_ASSET_PATH if WEIGHTS_ASSET_PATH.parent.is_dir() else APP_ASSET_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MobileCLIP asset to %s …", dest)
    attempt_download_asset(str(dest))
    if not _is_loadable(dest):
        raise RuntimeError(f"MobileCLIP download failed or asset still corrupt: {dest}")
    logger.info("MobileCLIP asset ready: %s", dest)
    _link_app_asset(dest)
    return dest


def _link_app_asset(source: Path) -> None:
    """Make /app/mobileclip_blt.ts point at the validated file for Ultralytics lookup."""
    if _is_loadable(APP_ASSET_PATH):
        return
    if source.resolve() == APP_ASSET_PATH.resolve():
        return
    if APP_ASSET_PATH.is_symlink():
        APP_ASSET_PATH.unlink()
    elif APP_ASSET_PATH.exists():
        return  # e.g. bind-mounted file — cannot replace
    APP_ASSET_PATH.symlink_to(source)
