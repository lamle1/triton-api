"""
gpu_manager.py — Discover GPUs visible to the API container and validate assignments.

Triton instance_group gpus: [N] must reference indices exposed inside the container
(typically 0 .. device_count-1 when NVIDIA_VISIBLE_DEVICES=all).
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

_gpu_cache: Optional[list[dict]] = None


def discover_gpus(refresh: bool = False) -> list[dict]:
    """
    Return GPUs available for Triton instance_group assignment.

    Each entry:
      { "index": int, "name": str, "memory_total_mb": int | null }
    """
    global _gpu_cache
    if _gpu_cache is not None and not refresh:
        return list(_gpu_cache)

    gpus: list[dict] = []

    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    "index": i,
                    "name": props.name,
                    "memory_total_mb": int(props.total_memory // (1024 * 1024)),
                })
    except Exception as exc:
        logger.warning(f"torch GPU discovery failed: {exc}")

    if not gpus:
        gpus = _discover_via_nvidia_smi()

    _gpu_cache = gpus
    return list(gpus)


def _discover_via_nvidia_smi() -> list[dict]:
    """Fallback when torch CUDA is unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        logger.warning(f"nvidia-smi unavailable: {exc}")
        return []

    gpus: list[dict] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        idx = int(parts[0])
        name = parts[1]
        mem_mb: Optional[int] = None
        if len(parts) >= 3 and parts[2].isdigit():
            mem_mb = int(parts[2])
        gpus.append({"index": idx, "name": name, "memory_total_mb": mem_mb})
    return gpus


def valid_gpu_indices(refresh: bool = False) -> set[int]:
    return {g["index"] for g in discover_gpus(refresh=refresh)}


def validate_instance_groups(groups: list[dict], refresh: bool = False) -> None:
    """
    Raise ValueError if any KIND_GPU group references an invalid GPU index.
    """
    valid = valid_gpu_indices(refresh=refresh)
    if not valid:
        logger.warning("No GPUs detected — skipping instance_group validation")
        return

    for g in groups:
        kind = g.get("kind", "KIND_GPU")
        if kind != "KIND_GPU":
            continue
        for gpu_id in g.get("gpus", []):
            idx = int(gpu_id)
            if idx not in valid:
                raise ValueError(
                    f"GPU index {idx} is not available. "
                    f"Valid indices: {sorted(valid)}. "
                    f"Use GET /gpus for the full list."
                )


def models_per_gpu(model_repo: str) -> dict[str, list[str]]:
    """
    Map GPU index (as string) → model names using each model's config.pbtxt.
    """
    mapping: dict[str, list[str]] = {}
    if not os.path.isdir(model_repo):
        return mapping

    for name in os.listdir(model_repo):
        cfg_path = os.path.join(model_repo, name, "config.pbtxt")
        if not os.path.isfile(cfg_path):
            continue
        with open(cfg_path) as f:
            text = f.read()
        if re.search(r'platform:\s*"ensemble"', text):
            continue
        ig = re.search(
            r"instance_group\s*\[.*?gpus:\s*\[\s*([^\]]+)\s*\]",
            text,
            re.DOTALL,
        )
        if not ig:
            continue
        for part in ig.group(1).split(","):
            part = part.strip()
            if not part:
                continue
            key = str(int(part))
            if name not in mapping.get(key, []):
                mapping.setdefault(key, []).append(name)

    return mapping


def default_gpu_index() -> int:
    """First GPU index, or 0 if discovery fails."""
    gpus = discover_gpus()
    return int(gpus[0]["index"]) if gpus else int(os.getenv("DEFAULT_GPU", "0"))
