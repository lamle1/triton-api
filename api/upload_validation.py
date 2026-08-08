"""
upload_validation.py — Pre-flight checks for model upload.
"""
from __future__ import annotations

import os
from typing import Optional

from inference_utils import APIError, default_model_name_from_filename, validate_model_name

MIN_PT_BYTES = 1024
MIN_ONNX_BYTES = 128
MAX_PT_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB


def resolve_upload_name(
    name: Optional[str],
    filename: Optional[str],
) -> str:
    if name and name.strip():
        return validate_model_name(name)
    return default_model_name_from_filename(filename)


def validate_pt_upload(
    pt_bytes: bytes,
    filename: Optional[str],
) -> None:
    if not pt_bytes:
        raise APIError("Uploaded file is empty")

    if len(pt_bytes) < MIN_PT_BYTES:
        raise APIError(f"File too small to be a valid .pt weights file (< {MIN_PT_BYTES} bytes)")

    if len(pt_bytes) > MAX_PT_BYTES:
        raise APIError(f"File exceeds maximum upload size ({MAX_PT_BYTES // (1024**3)} GiB)")

    if filename:
        ext = os.path.splitext(filename)[1].lower()
        if ext and ext not in (".pt", ".pth"):
            raise APIError("Only .pt or .pth weight files are supported")


def validate_model_upload(
    model_bytes: bytes,
    filename: Optional[str],
) -> str:
    """Validate upload envelope and return normalized extension: '.pt', '.pth', or '.onnx'."""
    if not model_bytes:
        raise APIError("Uploaded file is empty")
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in (".pt", ".pth", ".onnx"):
        raise APIError("Only .pt, .pth, or .onnx model files are supported")
    if ext == ".onnx":
        if len(model_bytes) < MIN_ONNX_BYTES:
            raise APIError(f"File too small to be a valid .onnx model (< {MIN_ONNX_BYTES} bytes)")
        if len(model_bytes) > MAX_PT_BYTES:
            raise APIError(f"File exceeds maximum upload size ({MAX_PT_BYTES // (1024**3)} GiB)")
        return ".onnx"
    validate_pt_upload(model_bytes, filename)
    return ext


def check_model_repo_slot(
    model_repo: str,
    model_name: str,
    overwrite: bool,
) -> None:
    model_dir = os.path.join(model_repo, model_name)
    if os.path.isdir(model_dir):
        if not overwrite:
            raise APIError(
                f"Model '{model_name}' already exists in the repository. "
                "Delete it first or pass overwrite=true.",
                status_code=409,
            )


def validate_pt_loadable(pt_path: str) -> None:
    """Quick check that the file is a loadable PyTorch checkpoint."""
    try:
        import torch

        try:
            ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(pt_path, map_location="cpu")
    except Exception as exc:
        raise APIError(f"Invalid or corrupt .pt file: {exc}") from exc

    if not isinstance(ckpt, dict):
        raise APIError("File does not look like an ultralytics/YOLO checkpoint")
