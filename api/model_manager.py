"""
model_manager.py — Model upload, conversion, deletion, and Triton hot-reload.

Upload pipeline
───────────────
1. Save .pt to a temp directory
2. Export to ONNX via ultralytics (runs in a thread-pool executor so
   it does not block the async event loop — can take 30-60 s)
3. Auto-detect model type (yoloe-dynamic / yolo, detect / seg / pose)
4. Generate a base config.pbtxt, merge any client overrides
5. Write ONNX + config.pbtxt into model_repo/{name}/1/
6. POST triton:8000/v2/repository/models/{name}/load  → hot-load

Delete pipeline
───────────────
1. POST triton:8000/v2/repository/models/{name}/unload
2. shutil.rmtree  model_repo/{name}
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
from typing import Optional

import httpx

from config_manager import generate_base_config, parse_config
from inference_utils import APIError, DEFAULT_IMGSZ, parse_imgsz
from model_detector import MODEL_META_FILE, detect_model_type
from model_detector import ONNXCompatibilityError, validate_and_normalize_onnx
from upload_validation import (
    check_model_repo_slot,
    validate_model_upload,
    validate_pt_loadable,
    validate_pt_upload,
)

logger = logging.getLogger(__name__)

# Triton management API paths
_LOAD_PATH = "/v2/repository/models/{name}/load"
_UNLOAD_PATH = "/v2/repository/models/{name}/unload"
_INDEX_PATH = "/v2/repository/index"


# ──────────────────────── Triton HTTP helpers ────────────────────

async def triton_load_model(http_url: str, model_name: str) -> None:
    """Signal Triton to load (or reload) a model by name."""
    base_url = http_url.rstrip("/")
    async with httpx.AsyncClient() as client:
        try:
            await client.post(base_url + _INDEX_PATH, timeout=5.0)
        except Exception:
            pass
        url = base_url + _LOAD_PATH.format(name=model_name)
        resp = await client.post(url, timeout=120.0)
        if resp.status_code == 503:
            # poll mode: model loads on next repository poll
            logger.warning(
                f"Triton load returned 503 for {model_name} "
                "(poll mode — will appear after repository poll)"
            )
            return
        if resp.status_code != 200:
            err_msg = resp.text
            try:
                err_json = resp.json()
                if isinstance(err_json, dict) and "error" in err_json:
                    err_msg = err_json["error"]
            except Exception:
                pass
            raise RuntimeError(f"Triton load failed for '{model_name}': {err_msg}")
    logger.info(f"Triton loaded model: {model_name}")


async def triton_unload_model(http_url: str, model_name: str) -> None:
    """Signal Triton to unload a model.  HTTP 400 (not loaded) is tolerated."""
    url = http_url.rstrip("/") + _UNLOAD_PATH.format(name=model_name)
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, timeout=30.0)
        if resp.status_code not in (200, 400):
            resp.raise_for_status()
    logger.info(f"Triton unloaded model: {model_name}")


async def list_models(http_url: str) -> list:
    """Return Triton repository index (list of dicts with 'name', 'state', etc.)."""
    url = http_url.rstrip("/") + _INDEX_PATH
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, timeout=15.0)   # POST returns full index
        resp.raise_for_status()
    return resp.json()


def split_models_by_kind(model_repo: str, triton_index: list) -> dict:
    """
    Partition Triton repository index into single (ONNX) vs ensemble models.

    Each entry is enriched with:
      kind     — "single" | "ensemble"
      platform — from config.pbtxt when present (e.g. onnxruntime_onnx, ensemble)
    """
    from ensemble_manager import is_ensemble
    from config_manager import read_model_config

    from ensemble_manager import HYBRID_META_FILE, get_ensemble_kind

    by_name: dict[str, dict] = {}
    for entry in triton_index:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not name or name in by_name:
            continue

        ens_kind = get_ensemble_kind(model_repo, name)
        kind = "ensemble" if ens_kind else "single"
        platform: str | None = None
        cfg_path = os.path.join(model_repo, name, "config.pbtxt")
        if os.path.isfile(cfg_path):
            try:
                platform = read_model_config(model_repo, name).get("platform")
            except Exception:
                pass

        row = {
            **entry,
            "kind": kind,
            "platform": platform or ("ensemble" if kind == "ensemble" else None),
        }
        if ens_kind:
            row["ensemble_kind"] = ens_kind  # "native" | "hybrid"
        by_name[name] = row

    # Hybrid ensembles are API-only (not in Triton index).
    if os.path.isdir(model_repo):
        for name in os.listdir(model_repo):
            meta = os.path.join(model_repo, name, HYBRID_META_FILE)
            if name in by_name or not os.path.isfile(meta):
                continue
            by_name[name] = {
                "name": name,
                "version": "1",
                "state": "READY",
                "kind": "ensemble",
                "platform": "hybrid",
                "ensemble_kind": "hybrid",
            }

    all_models = list(by_name.values())
    single_models = [m for m in all_models if m["kind"] == "single"]
    ensemble_models = [m for m in all_models if m["kind"] == "ensemble"]
    return {
        "models": all_models,
        "single_models": single_models,
        "ensemble_models": ensemble_models,
    }


# ──────────────────────── ONNX conversion ────────────────────────

def _export_to_onnx_sync(
    pt_path: str,
    output_dir: str,
    dynamic: bool = True,
    yoloe_dynamic: bool = False,
    imgsz: int = DEFAULT_IMGSZ,
) -> str:
    """
    Synchronous ONNX export (blocking — run via run_in_executor).

    YOLOE + yoloe_dynamic=True → two-input ONNX (images + prompt_embedding).
    Other weights → standard ultralytics export (single input).
    """
    from yoloe_export import export_yoloe_dynamic_sync, is_yoloe_checkpoint

    if yoloe_dynamic and is_yoloe_checkpoint(pt_path):
        logger.info("YOLOE detected — exporting dynamic two-input ONNX (imgsz=%s)", imgsz)
        try:
            return export_yoloe_dynamic_sync(pt_path, output_dir, imgsz=imgsz)
        except Exception as exc:
            if isinstance(exc, APIError):
                raise
            raise APIError(f"YOLOE ONNX export failed: {exc}", status_code=422) from exc

    from ultralytics import YOLO

    model = YOLO(pt_path)
    try:
        onnx_path = model.export(
            format="onnx",
            dynamic=dynamic,
            simplify=True,
            imgsz=imgsz,
        )
    except Exception as exc:
        raise APIError(f"ONNX export failed: {exc}", status_code=422) from exc
    return str(onnx_path)


async def _export_to_onnx(
    pt_path: str,
    output_dir: str,
    dynamic: bool = True,
    yoloe_dynamic: bool = False,
    imgsz: int = DEFAULT_IMGSZ,
) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: _export_to_onnx_sync(pt_path, output_dir, dynamic, yoloe_dynamic, imgsz),
    )


# ──────────────────────────── upload ─────────────────────────────

async def _deploy_onnx_model(
    model_repo: str,
    triton_http_url: str,
    model_name: str,
    onnx_path: str,
    config_overrides: Optional[dict] = None,
    overwrite: bool = False,
    source_format: str = "onnx",
    export_imgsz: int | None = None,
    encoder_pt_path: str | None = None,
) -> dict:
    """Deploy a validated ONNX file into Triton model_repo and hot-load it."""
    check_model_repo_slot(model_repo, model_name, overwrite)
    try:
        onnx_signature = validate_and_normalize_onnx(onnx_path)
    except ONNXCompatibilityError as exc:
        raise APIError(f"ONNX model is not compatible with this API: {exc}", status_code=422) from exc

    model_dir = os.path.join(model_repo, model_name)
    if overwrite and os.path.isdir(model_dir):
        await triton_unload_model(triton_http_url, model_name)
        shutil.rmtree(model_dir)

    model_info = detect_model_type(onnx_path)
    model_info = {
        **model_info,
        "source_format": source_format,
        "adapter": onnx_signature.get("adapter"),
        "output0_layout": onnx_signature.get("output0_layout"),
        "input_names": onnx_signature.get("input_names", []),
        "output_names": onnx_signature.get("output_names", []),
        "compatibility": {
            "ok": True,
            "contract": "yolo_raw",
        },
    }

    os.makedirs(model_dir, exist_ok=True)
    if model_info["type"] == "yoloe-dynamic" and encoder_pt_path:
        shutil.copy2(encoder_pt_path, os.path.join(model_dir, "encoder.pt"))

    version_dir = os.path.join(model_dir, "1")
    os.makedirs(version_dir, exist_ok=True)
    dest_onnx = os.path.join(version_dir, "model.onnx")
    shutil.copy2(onnx_path, dest_onnx)

    config_text = generate_base_config(
        model_name=model_name,
        model_type=model_info["type"],
        task=model_info["task"],
        num_classes=model_info.get("num_classes"),
        overrides=config_overrides,
        input_dims=onnx_signature.get("image_dims"),
        output0_dims=onnx_signature.get("output0_dims"),
        output1_dims=onnx_signature.get("output1_dims"),
    )
    config_path = os.path.join(model_repo, model_name, "config.pbtxt")
    with open(config_path, "w") as f:
        f.write(config_text)
    with open(os.path.join(model_dir, MODEL_META_FILE), "w") as f:
        json.dump(
            {
                **model_info,
                "onnx_signature": onnx_signature,
            },
            f,
            indent=2,
        )

    try:
        await triton_load_model(triton_http_url, model_name)
    except httpx.HTTPStatusError as exc:
        try:
            await triton_unload_model(triton_http_url, model_name)
        except Exception:
            pass
        shutil.rmtree(model_dir, ignore_errors=True)
        raise APIError(
            f"Triton failed to load model '{model_name}': {exc.response.text}",
            status_code=502,
        ) from exc
    except Exception as exc:
        try:
            await triton_unload_model(triton_http_url, model_name)
        except Exception:
            pass
        shutil.rmtree(model_dir, ignore_errors=True)
        raise APIError(
            f"Triton failed to load model '{model_name}': {exc}",
            status_code=502,
        ) from exc

    applied_config = {
        k: v for k, v in parse_config(config_text).items()
        if not k.startswith("_")
    }
    result = {
        "status": "loaded",
        "model": model_name,
        "type": f"{model_info['type']}-{model_info['task']}",
        "source_format": source_format,
        "adapter": model_info.get("adapter"),
        "output0_layout": model_info.get("output0_layout"),
        "config": applied_config,
        "onnx_signature": onnx_signature,
    }
    if export_imgsz is not None:
        result["export_imgsz"] = export_imgsz
    return result

async def upload_model(
    model_repo: str,
    triton_http_url: str,
    model_name: str,
    pt_bytes: bytes,
    config_overrides: Optional[dict] = None,
    dynamic_export: bool = True,
    yoloe_dynamic_export: bool = False,
    export_imgsz: int = DEFAULT_IMGSZ,
    overwrite: bool = False,
    filename: Optional[str] = None,
) -> dict:
    """
    Full upload → convert → deploy pipeline.

    Returns a summary dict:
      { status, model, type, config (without private _keys) }
    """
    validate_pt_upload(pt_bytes, filename)
    check_model_repo_slot(model_repo, model_name, overwrite)

    with tempfile.TemporaryDirectory() as tmpdir:
        pt_path = os.path.join(tmpdir, f"{model_name}.pt")
        with open(pt_path, "wb") as f:
            f.write(pt_bytes)

        validate_pt_loadable(pt_path)

        logger.info(f"Exporting {model_name} to ONNX (imgsz={export_imgsz}) …")
        try:
            onnx_path = await _export_to_onnx(
                pt_path,
                tmpdir,
                dynamic=dynamic_export,
                yoloe_dynamic=yoloe_dynamic_export,
                imgsz=export_imgsz,
            )
        except APIError:
            raise
        except Exception as exc:
            raise APIError(f"ONNX export failed: {exc}", status_code=422) from exc
        logger.info(f"ONNX export done: {onnx_path}")

        return await _deploy_onnx_model(
            model_repo=model_repo,
            triton_http_url=triton_http_url,
            model_name=model_name,
            onnx_path=onnx_path,
            config_overrides=config_overrides,
            overwrite=overwrite,
            source_format="pt",
            export_imgsz=export_imgsz,
            encoder_pt_path=pt_path,
        )


async def upload_model_file(
    model_repo: str,
    triton_http_url: str,
    model_name: str,
    model_bytes: bytes,
    config_overrides: Optional[dict] = None,
    dynamic_export: bool = True,
    yoloe_dynamic_export: bool = False,
    export_imgsz: int = DEFAULT_IMGSZ,
    overwrite: bool = False,
    filename: Optional[str] = None,
) -> dict:
    """Upload either .pt/.pth or already-exported .onnx."""
    ext = validate_model_upload(model_bytes, filename)
    if ext == ".onnx":
        check_model_repo_slot(model_repo, model_name, overwrite)
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, f"{model_name}.onnx")
            with open(onnx_path, "wb") as f:
                f.write(model_bytes)
            return await _deploy_onnx_model(
                model_repo=model_repo,
                triton_http_url=triton_http_url,
                model_name=model_name,
                onnx_path=onnx_path,
                config_overrides=config_overrides,
                overwrite=overwrite,
                source_format="onnx",
            )

    return await upload_model(
        model_repo=model_repo,
        triton_http_url=triton_http_url,
        model_name=model_name,
        pt_bytes=model_bytes,
        config_overrides=config_overrides,
        dynamic_export=dynamic_export,
        yoloe_dynamic_export=yoloe_dynamic_export,
        export_imgsz=export_imgsz,
        overwrite=overwrite,
        filename=filename,
    )


# ─────────────────────────── delete ──────────────────────────────

async def delete_model(
    model_repo: str,
    triton_http_url: str,
    model_name: str,
) -> None:
    """Unload from Triton and remove from model_repo."""
    await triton_unload_model(triton_http_url, model_name)

    model_dir = os.path.join(model_repo, model_name)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        logger.info(f"Removed model directory: {model_dir}")
    else:
        logger.warning(f"Model directory not found: {model_dir}")
