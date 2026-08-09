"""
label_manager.py — Per-model class-label management.

Labels are stored as a JSON array at:
  {MODEL_REPO_PATH}/{model_name}/labels.json

An in-process dict cache avoids repeated disk reads on hot paths.
Call invalidate_labels_cache(name) after any write.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

# In-process cache: model_name → (labels.json mtime, labels)
_cache: dict[str, tuple[float, list[str]]] = {}


def _labels_path(model_repo: str, model_name: str) -> str:
    return os.path.join(model_repo, model_name, "labels.json")


def read_labels(model_repo: str, model_name: str) -> Optional[list[str]]:
    """
    Return the label list for *model_name*, or None if no labels.json exists.
    Results are cached in memory.
    """
    path = _labels_path(model_repo, model_name)
    if not os.path.exists(path):
        _cache.pop(model_name, None)
        return None
    mtime = os.path.getmtime(path)
    cached = _cache.get(model_name)
    if cached and cached[0] == mtime:
        return list(cached[1])

    with open(path) as f:
        labels: list[str] = json.load(f)

    _cache[model_name] = (mtime, labels)
    logger.debug(f"Loaded {len(labels)} labels for model '{model_name}'")
    return list(labels)


def write_labels(model_repo: str, model_name: str, labels: list[str]) -> None:
    """
    Persist *labels* to disk and update the in-process cache.
    Creates parent directories if missing.
    """
    if not isinstance(labels, list) or not all(isinstance(l, str) for l in labels):
        raise ValueError("labels must be a list of strings")

    path = _labels_path(model_repo, model_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    parent = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(prefix=".labels.", suffix=".json", dir=parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(labels, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    _cache[model_name] = (os.path.getmtime(path), list(labels))
    logger.info(f"Wrote {len(labels)} labels for model '{model_name}'")


def delete_labels(model_repo: str, model_name: str) -> bool:
    """Delete labels.json if present and clear cache. Returns True if a file was removed."""
    path = _labels_path(model_repo, model_name)
    _cache.pop(model_name, None)
    try:
        os.remove(path)
        logger.info(f"Deleted labels for model '{model_name}'")
        return True
    except FileNotFoundError:
        return False


def invalidate_labels_cache(model_name: str) -> None:
    """Remove the cached label list for *model_name*."""
    _cache.pop(model_name, None)
