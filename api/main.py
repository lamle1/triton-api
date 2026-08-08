"""
main.py — Vision Inference API Server
FastAPI + Uvicorn  |  port 8003

Client docs: USAGE.md (Deployment & API Guide), ARCHITECTURE.md (Technical Reference).
OpenAPI: /docs, /redoc.

Shared state (initialised once in lifespan):
  encoder   : YOLOETextEncoder  — text prompt → [1,N,512] embedding
  triton    : TritonGRPCClient  — gRPC to Triton :8001
  semaphore : asyncio.Semaphore(MAX_CONCURRENT) — caps Triton call concurrency
"""
from __future__ import annotations

import asyncio
import base64
import threading
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import sqlite3
import re
import traceback
import uuid
import cv2
import httpx
from tracker import BYTETracker
from reid_client import extract_embedding
from database import (
    search_person, add_person_event, init_db, purge_expired_data,
    search_object, search_object_with_meta, search_object_with_hits,
    list_tracked, list_classes, delete_tracked, add_object_event,
    get_object_embedding, get_trajectory, list_unique_sessions,
    update_object_event_trail, update_object_event_image, update_object_last_seen,
)

class FFmpegVideoWriter:
    def __init__(self, filepath, width, height, fps):
        self.filepath = filepath
        self.width = width
        self.height = height
        self.fps = fps
        self.process = None
        self.fallback_writer = None
        
        try:
            self.cmd = [
                "ffmpeg",
                "-y",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}",
                "-r", str(fps),
                "-i", "-",
                "-an",
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "ultrafast",
                "-tune", "zerolatency",
                "-movflags", "+faststart",
                filepath
            ]
            self.process = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except Exception as e:
            logging.error(f"[FFmpegVideoWriter] Failed to start ffmpeg, falling back to cv2.VideoWriter: {e}")
            self.process = None
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.fallback_writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

    def write(self, frame):
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(frame.tobytes())
            except Exception as e:
                logging.error(f"[FFmpegVideoWriter] Write failed, falling back: {e}")
                self.process = None
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.fallback_writer = cv2.VideoWriter(self.filepath, fourcc, self.fps, (self.width, self.height))
                self.fallback_writer.write(frame)
        elif self.fallback_writer:
            self.fallback_writer.write(frame)

    def release(self):
        if self.process:
            if self.process.stdin:
                try:
                    self.process.stdin.close()
                except Exception:
                    pass
            self.process.wait()
            self.process = None
        if self.fallback_writer:
            self.fallback_writer.release()
            self.fallback_writer = None


from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from urllib.parse import urlparse
from contextlib import asynccontextmanager
from typing import Any, Optional

import numpy as np

from config_manager import (
    get_instance_groups,
    read_model_config,
    update_instance_groups,
    update_model_config,
)
from encoder import YOLOETextEncoder
from ensemble_manager import (
    analyze_ensemble_steps,
    create_ensemble,
    delete_ensemble,
    get_ensemble_kind,
    is_ensemble,
    parse_ensemble_steps,
)
from system_status import collect_system_status, fetch_triton_metrics_raw
from label_manager import delete_labels, read_labels, write_labels
from model_detector import MODEL_META_FILE, MODEL_TYPE_YOLOE_DYNAMIC, detect_model_type
from inference_utils import (
    APIError,
    DEFAULT_IMGSZ,
    parse_imgsz,
    parse_text_prompts,
    validate_model_name,
)
from model_manager import (
    delete_model,
    list_models,
    split_models_by_kind,
    triton_load_model,
    triton_unload_model,
    upload_model_file,
)
from postprocess import decode_coco_rle_mask, nms_backend, postprocess, postprocess_ensemble
from preprocess import preprocess
from upload_validation import resolve_upload_name
from gpu_manager import discover_gpus, models_per_gpu, validate_instance_groups
from stream_manager import StreamRateLimiter
from triton_client import TritonGRPCClient

# ─────────────────────────── config ──────────────────────────────
TRITON_GRPC_URL = os.getenv("TRITON_GRPC_URL", "triton-remote:8001")
TRITON_HTTP_URL = os.getenv("TRITON_HTTP_URL", "http://triton-remote:8000")
TRITON_METRICS_URL = os.getenv("TRITON_METRICS_URL", "http://triton-remote:8002/metrics")
YOLOE_WEIGHTS   = os.getenv("YOLOE_WEIGHTS",   "/weights/yoloe-v8s-seg.pt")
MODEL_REPO_PATH = os.getenv("MODEL_REPO_PATH",  "/model_repo")
DEFAULT_CONF    = float(os.getenv("DEFAULT_CONF",   "0.25"))
DEFAULT_IOU     = float(os.getenv("DEFAULT_IOU",    "0.45"))
MAX_CONCURRENT  = int(os.getenv("MAX_CONCURRENT",   "32"))
MAX_FPS         = float(os.getenv("MAX_FPS",         "0"))
MAX_BATCH_FILES = int(os.getenv("MAX_BATCH_FILES",   "32"))
MODEL_FAILURE_THRESHOLD = int(os.getenv("MODEL_FAILURE_THRESHOLD", "3"))
MODEL_FAILURE_COOLDOWN_SEC = float(os.getenv("MODEL_FAILURE_COOLDOWN_SEC", "20"))
RTSP_OPEN_TIMEOUT_MS = int(os.getenv("RTSP_OPEN_TIMEOUT_MS", "5000"))
RTSP_READ_TIMEOUT_MS = int(os.getenv("RTSP_READ_TIMEOUT_MS", "5000"))
RTSP_JPEG_QUALITY    = int(os.getenv("RTSP_JPEG_QUALITY", "80"))
RTSP_INFER_MAX_WIDTH = int(os.getenv("RTSP_INFER_MAX_WIDTH", "1280"))
RTSP_INFER_MAX_HEIGHT = int(os.getenv("RTSP_INFER_MAX_HEIGHT", "720"))
RTSP_PREVIEW_MAX_WIDTH = int(os.getenv("RTSP_PREVIEW_MAX_WIDTH", "1280"))
RTSP_PREVIEW_MAX_HEIGHT = int(os.getenv("RTSP_PREVIEW_MAX_HEIGHT", "720"))
RTSP_MAX_INFLIGHT_INFER = max(1, int(os.getenv("RTSP_MAX_INFLIGHT_INFER", "1")))
RTSP_MAX_RESULT_AGE_MS = max(100, int(os.getenv("RTSP_MAX_RESULT_AGE_MS", "3000")))
RTSP_MAX_READ_FAILURES = int(os.getenv("RTSP_MAX_READ_FAILURES", "30"))
RTSP_TRANSPORT       = os.getenv("RTSP_TRANSPORT", "tcp").strip().lower()
RTSP_LOW_LATENCY     = os.getenv("RTSP_LOW_LATENCY", "true").strip().lower() not in {"0", "false", "no"}
RTSP_GRAB_LATEST_FRAMES = int(os.getenv("RTSP_GRAB_LATEST_FRAMES", "0"))
RTSP_BACKEND         = os.getenv("RTSP_BACKEND", "auto").strip().lower()
RTSP_GSTREAMER_DECODER = os.getenv("RTSP_GSTREAMER_DECODER", "auto").strip().lower()
RTSP_GSTREAMER_LATENCY = int(os.getenv("RTSP_GSTREAMER_LATENCY", "0"))
RTSP_GSTREAMER_PIPELINE = os.getenv("RTSP_GSTREAMER_PIPELINE", "").strip()
GO2RTC_API_URL = os.getenv("GO2RTC_API_URL", "").strip().rstrip("/")
GO2RTC_PUBLIC_URL = os.getenv("GO2RTC_PUBLIC_URL", "").strip().rstrip("/")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────── shared state ────────────────────────────
encoder: Optional[YOLOETextEncoder] = None
_encoders_by_model: dict[str, YOLOETextEncoder] = {}
_bad_model_encoders: set[str] = set()
triton: Optional[TritonGRPCClient] = None
semaphore: Optional[asyncio.Semaphore] = None

# In-memory model-type registry (avoids re-inspecting ONNX on every request).
# Each entry includes a signature made from model files' mtimes so direct disk
# edits are picked up without restarting the API process.
_model_registry: dict[str, dict] = {}
_model_failures: dict[str, dict[str, Any]] = {}
MODEL_TYPE_ENSEMBLE = "ensemble"


def _model_disabled_message(model_name: str) -> str | None:
    state = _model_failures.get(model_name)
    if not state:
        return None
    until = float(state.get("until") or 0)
    if until and until <= time.monotonic():
        _model_failures.pop(model_name, None)
        return None
    if not until:
        return None
    left = max(1, int(round(until - time.monotonic())))
    return f"Model '{model_name}' temporarily paused for {left}s after repeated inference failures: {state.get('last_error', '')}"


def _record_model_success(model_name: str) -> None:
    _model_failures.pop(model_name, None)


def _record_model_failure(model_name: str, exc: BaseException) -> None:
    state = _model_failures.setdefault(model_name, {"count": 0, "until": 0.0, "last_error": ""})
    state["count"] = int(state.get("count") or 0) + 1
    state["last_error"] = str(exc)
    if state["count"] >= max(1, MODEL_FAILURE_THRESHOLD):
        state["until"] = time.monotonic() + max(1.0, MODEL_FAILURE_COOLDOWN_SEC)
        logger.warning(
            "Model %s paused for %.1fs after %s consecutive inference failures: %s",
            model_name,
            MODEL_FAILURE_COOLDOWN_SEC,
            state["count"],
            exc,
        )


def _model_signature(model_name: str) -> tuple[tuple[str, float | None], ...]:
    base = os.path.join(MODEL_REPO_PATH, model_name)
    paths = (
        os.path.join(base, "config.pbtxt"),
        os.path.join(base, "ensemble.json"),
        os.path.join(base, MODEL_META_FILE),
        os.path.join(base, "1", "model.onnx"),
    )
    sig: list[tuple[str, float | None]] = []
    for path in paths:
        try:
            sig.append((os.path.basename(path), os.path.getmtime(path)))
        except OSError:
            sig.append((os.path.basename(path), None))
    return tuple(sig)


def _public_model_info(info: dict) -> dict:
    return {k: v for k, v in info.items() if not k.startswith("_")}


def _model_info(model_name: str) -> dict:
    sig = _model_signature(model_name)
    cached = _model_registry.get(model_name)
    if cached and cached.get("_signature") == sig:
        return _public_model_info(cached)

    if is_ensemble(MODEL_REPO_PATH, model_name):
        raw_steps = parse_ensemble_steps(MODEL_REPO_PATH, model_name)
        try:
            analysis = analyze_ensemble_steps(MODEL_REPO_PATH, raw_steps)
            steps = analysis["steps"]
        except Exception:
            steps = raw_steps
        info = {
            "type": MODEL_TYPE_ENSEMBLE,
            "task": "ensemble",
            "ensemble_kind": get_ensemble_kind(MODEL_REPO_PATH, model_name),
            "has_masks": any(s.get("has_masks") for s in steps),
            "steps": steps,
        }
        _model_registry[model_name] = {**info, "_signature": sig}
        return info

    onnx_path = os.path.join(MODEL_REPO_PATH, model_name, "1", "model.onnx")
    if os.path.exists(onnx_path):
        info = detect_model_type(onnx_path)
        meta_path = os.path.join(MODEL_REPO_PATH, model_name, MODEL_META_FILE)
        if os.path.isfile(meta_path):
            try:
                with open(meta_path) as f:
                    info = {**info, **json.load(f)}
            except Exception as exc:
                logger.warning("Could not read model metadata for %s: %s", model_name, exc)
        _model_registry[model_name] = {**info, "_signature": sig}
        return info

    return {"type": "yolo", "task": "detect", "has_masks": False, "num_classes": None}


def _model_exists_on_disk(model_name: str) -> bool:
    return os.path.isdir(os.path.join(MODEL_REPO_PATH, model_name))


def _disk_model_names(include_hybrid: bool = False) -> list[str]:
    if not os.path.isdir(MODEL_REPO_PATH):
        return []
    names: list[str] = []
    for name in sorted(os.listdir(MODEL_REPO_PATH)):
        try:
            validate_model_name(name)
        except Exception:
            continue
        model_dir = os.path.join(MODEL_REPO_PATH, name)
        if not os.path.isdir(model_dir):
            continue
        kind = get_ensemble_kind(MODEL_REPO_PATH, name)
        if kind == "hybrid" and not include_hybrid:
            continue
        if kind in {"native", "hybrid"} or os.path.isfile(os.path.join(model_dir, "1", "model.onnx")):
            names.append(name)
    return names


def _labels_count(model_name: str) -> int | None:
    labels = read_labels(MODEL_REPO_PATH, model_name)
    return len(labels) if labels is not None else None


def _public_failure_state(model_name: str) -> dict[str, Any] | None:
    state = _model_failures.get(model_name)
    if not state:
        return None
    until = float(state.get("until") or 0)
    return {
        "count": int(state.get("count") or 0),
        "paused": until > time.monotonic(),
        "paused_seconds_remaining": max(0, int(round(until - time.monotonic()))),
        "last_error": state.get("last_error", ""),
    }


async def _triton_index_by_name() -> dict[str, dict]:
    try:
        index = await list_models(TRITON_HTTP_URL)
    except Exception:
        return {}
    return {
        entry.get("name"): entry
        for entry in index
        if isinstance(entry, dict) and entry.get("name")
    }


async def _wait_triton_model_ready(model_name: str, timeout_s: float = 30.0) -> None:
    """Wait until Triton repository index reports a model as READY."""
    deadline = time.monotonic() + timeout_s
    last_state = "missing"
    while time.monotonic() < deadline:
        index = await _triton_index_by_name()
        entry = index.get(model_name)
        if entry and entry.get("state") == "READY":
            return
        last_state = str(entry.get("state") if entry else "missing")
        await asyncio.sleep(0.25)
    raise RuntimeError(f"Model '{model_name}' did not become READY in Triton (last_state={last_state})")


async def _sync_triton_models_from_disk() -> dict[str, Any]:
    """Load every valid disk model into Triton; tolerate empty repos and bad models."""
    await _cleanup_hybrid_ensemble_tensorrt_artifacts()
    names = _disk_model_names(include_hybrid=False)
    if not names:
        logger.info("No Triton-loadable models found on disk; API will start with empty model repo.")
        return {"loaded": [], "failed": {}, "skipped": []}

    loaded: list[str] = []
    failed: dict[str, str] = {}
    skipped: list[str] = []

    def sort_key(name: str) -> tuple[int, str]:
        return (1 if get_ensemble_kind(MODEL_REPO_PATH, name) == "native" else 0, name)

    for name in sorted(names, key=sort_key):
        try:
            if get_ensemble_kind(MODEL_REPO_PATH, name) == "native":
                missing = [
                    step.get("model")
                    for step in parse_ensemble_steps(MODEL_REPO_PATH, name)
                    if step.get("model") and not _model_exists_on_disk(step["model"])
                ]
                if missing:
                    skipped.append(name)
                    failed[name] = f"missing ensemble steps: {', '.join(missing)}"
                    continue
            await triton_load_model(TRITON_HTTP_URL, name)
            loaded.append(name)
        except Exception as exc:
            failed[name] = str(exc)
            logger.warning("Startup model load skipped for %s: %s", name, exc)

    logger.info("Startup Triton sync complete: loaded=%s failed=%s", loaded, failed)
    return {"loaded": loaded, "failed": failed, "skipped": skipped}


async def _cleanup_hybrid_ensemble_tensorrt_artifacts() -> None:
    """
    Hybrid ensembles are API-managed and must not look like Triton models.
    Older versions created an empty version folder (`1/`) next to ensemble.json,
    which caused Triton readiness to stay false after restart.
    """
    if not os.path.isdir(MODEL_REPO_PATH):
        return
    for name in os.listdir(MODEL_REPO_PATH):
        if get_ensemble_kind(MODEL_REPO_PATH, name) != "hybrid":
            continue
        version_dir = os.path.join(MODEL_REPO_PATH, name, "1")
        try:
            if os.path.isdir(version_dir) and not os.listdir(version_dir):
                shutil.rmtree(version_dir)
                logger.info("Removed empty Triton version dir from hybrid ensemble: %s", name)
                try:
                    await triton_unload_model(TRITON_HTTP_URL, name)
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("Could not clean hybrid ensemble Triton artifact %s: %s", name, exc)


async def _ensure_triton_has_disk_models() -> None:
    disk = set(_disk_model_names(include_hybrid=False))
    if not disk:
        return
    index = await _triton_index_by_name()
    ready_or_known = set(index.keys())
    if disk - ready_or_known:
        await _sync_triton_models_from_disk()


async def _effective_triton_ready() -> bool:
    """
    Triton `/v2/health/ready` can be false when the repository contains API-only
    hybrid ensemble metadata directories. Treat readiness as true when every
    Triton-loadable disk model is present and READY.
    """
    if triton and triton.is_server_ready():
        return True
    try:
        index = await list_models(TRITON_HTTP_URL)
    except Exception:
        return False
    loadable = set(_disk_model_names(include_hybrid=False))
    if not loadable:
        return True
    by_name = {e.get("name"): e for e in index if isinstance(e, dict) and e.get("name")}
    for name in loadable:
        if by_name.get(name, {}).get("state") != "READY":
            return False
    return True


async def _remove_model_from_dependent_ensembles(model_name: str) -> list[dict[str, Any]]:
    changed: list[dict[str, Any]] = []
    for ens_name in _disk_model_names(include_hybrid=True):
        if ens_name == model_name or not is_ensemble(MODEL_REPO_PATH, ens_name):
            continue
        try:
            steps = parse_ensemble_steps(MODEL_REPO_PATH, ens_name)
        except Exception as exc:
            logger.warning("Could not inspect ensemble %s while deleting %s: %s", ens_name, model_name, exc)
            continue
        if not any(step.get("model") == model_name for step in steps):
            continue

        remaining = [step for step in steps if step.get("model") != model_name]
        kind = get_ensemble_kind(MODEL_REPO_PATH, ens_name)
        try:
            if kind == "native":
                await triton_unload_model(TRITON_HTTP_URL, ens_name)
            if not remaining:
                delete_ensemble(MODEL_REPO_PATH, ens_name)
                changed.append({"ensemble": ens_name, "action": "deleted", "removed_model": model_name})
            else:
                create_ensemble(MODEL_REPO_PATH, ens_name, remaining)
                if get_ensemble_kind(MODEL_REPO_PATH, ens_name) == "native":
                    await triton_load_model(TRITON_HTTP_URL, ens_name)
                changed.append({
                    "ensemble": ens_name,
                    "action": "updated",
                    "removed_model": model_name,
                    "remaining_steps": [s.get("model") for s in remaining],
                })
            _model_registry.pop(ens_name, None)
        except Exception as exc:
            logger.warning("Failed to update ensemble %s after deleting %s: %s", ens_name, model_name, exc)
            changed.append({"ensemble": ens_name, "action": "error", "error": str(exc)})
    return changed


async def _rename_model_in_dependent_ensembles(old_name: str, new_name: str) -> list[dict[str, Any]]:
    changed: list[dict[str, Any]] = []
    for ens_name in _disk_model_names(include_hybrid=True):
        if ens_name in (old_name, new_name) or not is_ensemble(MODEL_REPO_PATH, ens_name):
            continue
        try:
            steps = parse_ensemble_steps(MODEL_REPO_PATH, ens_name)
        except Exception as exc:
            logger.warning("Could not inspect ensemble %s while renaming %s: %s", ens_name, old_name, exc)
            continue
        if not any(step.get("model") == old_name for step in steps):
            continue

        updated_steps = []
        for step in steps:
            s = dict(step)
            if s.get("model") == old_name:
                s["model"] = new_name
            updated_steps.append(s)

        kind = get_ensemble_kind(MODEL_REPO_PATH, ens_name)
        try:
            if kind == "native":
                await triton_unload_model(TRITON_HTTP_URL, ens_name)
            create_ensemble(MODEL_REPO_PATH, ens_name, updated_steps)
            if get_ensemble_kind(MODEL_REPO_PATH, ens_name) == "native":
                await triton_load_model(TRITON_HTTP_URL, ens_name)
            changed.append({
                "ensemble": ens_name,
                "action": "renamed_step",
                "old_model": old_name,
                "new_model": new_name,
            })
            _model_registry.pop(ens_name, None)
        except Exception as exc:
            logger.warning("Failed to update ensemble %s after renaming %s: %s", ens_name, old_name, exc)
            changed.append({"ensemble": ens_name, "action": "error", "error": str(exc)})
    return changed


def _round_timing(timing: dict[str, float]) -> dict[str, float]:
    return {k: round(float(v), 3) for k, v in timing.items()}


def _validate_rtsp_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"rtsp", "rtsps"} or not parsed.netloc:
        raise ValueError("RTSP URL must start with rtsp:// or rtsps:// and include a host")
    return url


def _clamp_rtsp_fps(fps: float) -> float:
    if fps <= 0:
        return float(MAX_FPS) if MAX_FPS > 0 else 0.0
    if MAX_FPS <= 0:
        return max(0.1, float(fps))
    return max(0.1, min(float(fps), float(MAX_FPS)))


def _clamp_jpeg_quality(quality: int) -> int:
    return max(30, min(int(quality), 95))


_STREAM_SOURCE_HEIGHT_PRESETS = {0, 512, 640, 720, 960, 1080, 1440}

# ByteTrack path-trail constants (read once at startup)
_TRAIL_MAX   = int(os.getenv("TRAIL_MAX_PTS", "60"))   # max bboxes to keep per track
_TRAIL_MINPX = float(os.getenv("TRAIL_MIN_PX", "3.0")) # min center-movement to record a new point


def _normalize_stream_source_max_height(value: int | str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except Exception:
        raise HTTPException(400, "source_max_height must be one of native, 512, 640, 720, 960, 1080, 1440")
    if parsed <= 0:
        return None
    if parsed not in _STREAM_SOURCE_HEIGHT_PRESETS:
        raise HTTPException(400, "source_max_height must be one of native, 512, 640, 720, 960, 1080, 1440")
    return parsed


def _has_nvidia_gpu() -> bool:
    visible = os.getenv("NVIDIA_VISIBLE_DEVICES", "").strip().lower()
    if visible and visible not in {"none", "void", "no", "0"}:
        return True
    if os.path.exists("/proc/driver/nvidia/version"):
        return True
    try:
        return bool(discover_gpus())
    except Exception:
        return False


def _opencv_has_gstreamer() -> bool:
    try:
        build = cv2.getBuildInformation()
    except Exception:
        return False
    return "GStreamer:                   YES" in build or "GStreamer: YES" in build


def _python_gstreamer_available() -> bool:
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        return True
    except Exception:
        return False


def _gstreamer_available() -> bool:
    return _opencv_has_gstreamer() or _python_gstreamer_available()


def _rtsp_effective_backend(requested: str | None = None) -> str:
    req = (requested or RTSP_BACKEND).strip().lower()
    backend = req if req in {"auto", "opencv", "ffmpeg", "gstreamer"} else "auto"
    if backend in {"opencv", "ffmpeg"}:
        return "opencv"
    if backend == "gstreamer":
        return "gstreamer" if _gstreamer_available() else "opencv"
    return "gstreamer" if _has_nvidia_gpu() and _gstreamer_available() else "opencv"


def _validate_stream_backend(backend: str | None) -> str:
    value = (backend or RTSP_BACKEND or "auto").strip().lower()
    if value not in {"auto", "opencv", "ffmpeg", "gstreamer"}:
        raise HTTPException(400, "backend must be one of auto, opencv, ffmpeg, gstreamer")
    return value


def _gstreamer_decoder_chain() -> str:
    decoder = _gstreamer_selected_decoder()
    if decoder in {"nvidia", "gpu", "nv"} and _gst_element_available("nvv4l2decoder"):
        # nvv4l2decoder is common on NVIDIA Jetson images.
        return "rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert"
    if decoder in {"nvh264dec", "nvdec"} and _gst_element_available("nvh264dec"):
        return "rtph264depay ! h264parse ! nvh264dec ! videoconvert"
    return "rtph264depay ! h264parse ! avdec_h264 ! videoconvert"


def _gstreamer_selected_decoder() -> str:
    decoder = RTSP_GSTREAMER_DECODER
    if decoder == "auto":
        if _has_nvidia_gpu() and _gst_element_available("nvv4l2decoder"):
            decoder = "nvidia"
        elif _has_nvidia_gpu() and _gst_element_available("nvh264dec"):
            decoder = "nvh264dec"
        else:
            decoder = "software"
    return decoder


_gst_element_cache: dict[str, bool] = {}


def _gst_element_available(name: str) -> bool:
    if name in _gst_element_cache:
        return _gst_element_cache[name]
    try:
        result = subprocess.run(
            ["gst-inspect-1.0", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )
        ok = result.returncode == 0
    except Exception:
        ok = False
    _gst_element_cache[name] = ok
    return ok


def _gstreamer_pipeline(url: str) -> str:
    if RTSP_GSTREAMER_PIPELINE:
        return RTSP_GSTREAMER_PIPELINE.format(
            url=url,
            transport=RTSP_TRANSPORT,
            latency=RTSP_GSTREAMER_LATENCY,
        )
    protocols = "tcp" if RTSP_TRANSPORT == "tcp" else "udp"
    return (
        f'rtspsrc location="{url}" protocols={protocols} latency={RTSP_GSTREAMER_LATENCY} '
        f"drop-on-latency=true ! {_gstreamer_decoder_chain()} ! "
        "video/x-raw,format=BGR ! appsink name=sink drop=true max-buffers=1 sync=false"
    )


class GstRTSPCapture:
    def __init__(self, url: str):
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        self.Gst = Gst
        pipeline = _gstreamer_pipeline(url)
        self.pipeline = Gst.parse_launch(pipeline)
        self.appsink = self.pipeline.get_by_name("sink")
        if self.appsink is None:
            raise RuntimeError("GStreamer pipeline missing appsink name=sink")
        self.appsink.set_property("emit-signals", False)
        self.appsink.set_property("sync", False)
        self.appsink.set_property("drop", True)
        self.appsink.set_property("max-buffers", 1)
        self.pipeline.set_state(Gst.State.PLAYING)
        ret, _, _ = self.pipeline.get_state(RTSP_OPEN_TIMEOUT_MS * Gst.MSECOND)
        if ret == Gst.StateChangeReturn.FAILURE:
            self.release()
            raise RuntimeError("Could not start GStreamer RTSP pipeline")
        self._last_frame = None

    def isOpened(self) -> bool:
        return self.pipeline is not None

    def read(self):
        sample = self.appsink.emit("try-pull-sample", RTSP_READ_TIMEOUT_MS * self.Gst.MSECOND)
        if sample is None:
            return False, None
        caps = sample.get_caps()
        struct = caps.get_structure(0)
        width = int(struct.get_value("width"))
        height = int(struct.get_value("height"))
        buf = sample.get_buffer()
        ok, info = buf.map(self.Gst.MapFlags.READ)
        if not ok:
            return False, None
        try:
            frame = np.frombuffer(info.data, dtype=np.uint8).reshape((height, width, 3)).copy()
        finally:
            buf.unmap(info)
        self._last_frame = frame
        return True, frame

    def grab(self) -> bool:
        ok, frame = self.read()
        if ok:
            self._last_frame = frame
        return ok

    def retrieve(self):
        return (self._last_frame is not None), self._last_frame

    def get(self, prop):
        return 0.0

    def release(self):
        if self.pipeline is not None:
            self.pipeline.set_state(self.Gst.State.NULL)
            self.pipeline = None


class BackgroundOpenCVCapture:
    """
    Wraps cv2.VideoCapture to run in a background daemon thread.
    This constantly drains the RTSP stream at its native frame rate, preventing
    FFmpeg/OpenCV from buffering stale frames when the main inference loop sleeps.
    """
    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.lock = threading.Lock()
        self.running = True
        self.last_frame = None
        self.last_ok = False
        self.exception = None
        self.thread = threading.Thread(target=self._reader_thread, daemon=True)
        self.thread.start()

    def _reader_thread(self):
        while self.running:
            try:
                ok, frame = self.cap.read()
                if not ok:
                    with self.lock:
                        self.last_ok = False
                        self.last_frame = None
                    time.sleep(0.01)
                    continue
                with self.lock:
                    self.last_ok = True
                    self.last_frame = frame
            except Exception as e:
                self.exception = e
                break

    def isOpened(self) -> bool:
        return self.cap.isOpened() and self.running

    def read(self):
        if self.exception:
            raise self.exception
        with self.lock:
            return self.last_ok, self.last_frame

    def grab(self) -> bool:
        ok, _ = self.read()
        return ok

    def retrieve(self):
        return self.read()

    def get(self, prop) -> float:
        return self.cap.get(prop)

    def release(self):
        self.running = False
        self.cap.release()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


def _open_rtsp_capture_opencv(url: str) -> BackgroundOpenCVCapture:
    ffmpeg_opts: list[str] = [
        f"stimeout;{RTSP_OPEN_TIMEOUT_MS * 1000}",
        f"timeout;{RTSP_OPEN_TIMEOUT_MS * 1000}"
    ]
    if RTSP_TRANSPORT in {"tcp", "udp"}:
        ffmpeg_opts.append(f"rtsp_transport;{RTSP_TRANSPORT}")
    if RTSP_LOW_LATENCY:
        ffmpeg_opts.extend([
            "fflags;nobuffer",
            "flags;low_delay",
        ])
    if ffmpeg_opts:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(ffmpeg_opts)
    params: list[int] = []
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        params.extend([int(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC), RTSP_OPEN_TIMEOUT_MS])
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        params.extend([int(cv2.CAP_PROP_READ_TIMEOUT_MSEC), RTSP_READ_TIMEOUT_MS])
    try:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, params) if params else cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    except TypeError:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_OPEN_TIMEOUT_MS)
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, RTSP_READ_TIMEOUT_MS)
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        # 3 frames: enough to absorb keyframe decode spikes (~30-80 ms on H.264 I-frames)
        # without building up latency. 1 caused the reader thread to stall on every GOP.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open RTSP stream")
    return BackgroundOpenCVCapture(cap)


def _open_rtsp_capture_gstreamer(url: str) -> cv2.VideoCapture:
    if not _opencv_has_gstreamer() and _python_gstreamer_available():
        return GstRTSPCapture(url)
    pipeline = _gstreamer_pipeline(url)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open RTSP stream with GStreamer")
    return cap


def _open_rtsp_capture(url: str, backend_requested: str | None = None) -> cv2.VideoCapture:
    backend = _rtsp_effective_backend(backend_requested)
    if backend == "gstreamer":
        try:
            logger.info("Opening RTSP with GStreamer backend")
            return _open_rtsp_capture_gstreamer(url)
        except Exception as exc:
            if (backend_requested or RTSP_BACKEND) == "gstreamer":
                raise
            logger.warning(f"GStreamer RTSP open failed, falling back to OpenCV/FFmpeg: {exc}")
    logger.info("Opening RTSP with OpenCV/FFmpeg backend")
    return _open_rtsp_capture_opencv(url)


def _read_rtsp_frame(cap: cv2.VideoCapture) -> tuple[np.ndarray, tuple[int, int]]:
    frame = None
    ok = False
    if RTSP_GRAB_LATEST_FRAMES > 0:
        for _ in range(RTSP_GRAB_LATEST_FRAMES):
            ok = cap.grab()
            if not ok:
                break
        if ok:
            ok, frame = cap.retrieve()
    if not ok or frame is None:
        ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Could not read RTSP frame")
    h, w = frame.shape[:2]
    return frame, (h, w)


def _read_rtsp_jpeg(cap: cv2.VideoCapture, quality: int) -> tuple[bytes, tuple[int, int]]:
    frame, shape = _read_rtsp_frame(cap)
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), _clamp_jpeg_quality(quality)],
    )
    if not ok:
        raise RuntimeError("Could not encode RTSP frame as JPEG")
    return encoded.tobytes(), shape


def _decode_jpeg_frame(jpeg: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Could not decode JPEG frame")
    return frame


def _annotation_color_key(ann: dict) -> str:
    """Match web client annColorKey(): one stable color per label (and model)."""
    source = ann.get("source_model") or ""
    name = ann.get("category_name") or f"class_{ann.get('category_id', '?')}"
    return f"{source}:{name}"


_PREVIEW_MASK_ALPHA = 90 / 255.0  # match web client drawAnnotations()


def _preview_color_map(annotations: list[dict]) -> dict[str, tuple[int, int, int]]:
    colors = [
        (74, 222, 128), (96, 165, 250), (248, 113, 113), (251, 191, 36),
        (167, 139, 250), (52, 211, 153), (244, 114, 182), (251, 146, 60),
    ]
    color_by_key: dict[str, tuple[int, int, int]] = {}
    for ann in annotations or []:
        key = _annotation_color_key(ann)
        if key not in color_by_key:
            color_by_key[key] = colors[len(color_by_key) % len(colors)]
    return color_by_key


def _draw_preview_masks(
    frame: np.ndarray,
    annotations: list[dict],
    color_by_key: dict[str, tuple[int, int, int]],
) -> np.ndarray:
    out = frame
    fh, fw = out.shape[:2]
    alpha = _PREVIEW_MASK_ALPHA
    inv = 1.0 - alpha

    for ann in annotations or []:
        seg = ann.get("segmentation")
        if not isinstance(seg, dict) or not seg.get("counts"):
            continue
        mask = decode_coco_rle_mask(seg)
        if mask is None:
            continue
        mh, mw = mask.shape[:2]
        if (mh, mw) != (fh, fw):
            mask = cv2.resize(
                mask.astype(np.uint8), (fw, fh), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        color = color_by_key.get(_annotation_color_key(ann), (74, 222, 128))
        for c in range(3):
            channel = out[:, :, c]
            channel[mask] = (
                channel[mask].astype(np.float32) * inv + color[c] * alpha
            ).astype(np.uint8)
            out[:, :, c] = channel
    return out


def _resize_frame_to_limits(
    frame: np.ndarray,
    max_width: int,
    max_height: int,
) -> tuple[np.ndarray, float]:
    max_w = max(0, int(max_width))
    max_h = max(0, int(max_height))
    if max_w <= 0 and max_h <= 0:
        return frame, 1.0
    h, w = frame.shape[:2]
    scale_w = max_w / w if max_w > 0 and w > max_w else 1.0
    scale_h = max_h / h if max_h > 0 and h > max_h else 1.0
    scale = min(scale_w, scale_h, 1.0)
    if scale >= 0.999:
        return frame, 1.0
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA), scale


def _resize_infer_frame(frame: np.ndarray) -> tuple[np.ndarray, float]:
    return _resize_frame_to_limits(frame, RTSP_INFER_MAX_WIDTH, RTSP_INFER_MAX_HEIGHT)


def _resize_preview_frame(frame: np.ndarray) -> tuple[np.ndarray, float]:
    return _resize_frame_to_limits(frame, RTSP_PREVIEW_MAX_WIDTH, RTSP_PREVIEW_MAX_HEIGHT)


def _draw_preview_annotations(
    frame: np.ndarray,
    annotations: list[dict],
    scale: float = 1.0,
) -> np.ndarray:
    ordered = list(annotations or [])
    color_by_key = _preview_color_map(ordered)
    out = _draw_preview_masks(frame.copy(), ordered, color_by_key)

    for ann in ordered:
        bbox = ann.get("bbox") or []
        if len(bbox) != 4:
            continue
        x, y, w, h = [int(round(float(v) * scale)) for v in bbox]
        color = color_by_key.get(_annotation_color_key(ann), (74, 222, 128))
        label = ann.get("category_name") or f"class_{ann.get('category_id', '?')}"
        score = ann.get("score")
        # Track ID is used in the background but hidden from live stream overlays for a cleaner UI
        text = f"{label} "
        if isinstance(score, (int, float)):
            text += f"{score * 100:.0f}%"
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 0), 5)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        ly = max(0, y - th - 10)
        cv2.rectangle(out, (x, ly), (x + tw + 12, ly + th + 10), (0, 0, 0), -1)
        cv2.rectangle(out, (x, ly), (x + tw + 12, ly + th + 10), color, -1)
        cv2.putText(out, text, (x + 6, ly + th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (5, 5, 5), 2, cv2.LINE_AA)
    return out


def _encode_jpeg_frame(frame: np.ndarray, quality: int) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), _clamp_jpeg_quality(quality)],
    )
    if not ok:
        raise RuntimeError("Could not encode preview frame")
    return encoded.tobytes()


def _queue_latest(q: asyncio.Queue, item: Any) -> None:
    try:
        if q.full():
            q.get_nowait()
    except asyncio.QueueEmpty:
        pass
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        pass


def _go2rtc_available() -> bool:
    return bool(GO2RTC_API_URL)


def _go2rtc_stream_name(stream_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"triton_{stream_id}")[:64]


def _go2rtc_public_base(request: Request | None = None) -> str | None:
    if GO2RTC_PUBLIC_URL:
        url = GO2RTC_PUBLIC_URL.rstrip("/")
        if ":1984" in url or url.endswith("/go2rtc"):
            return url
        return f"{url}/go2rtc"
    if request is None:
        return None
    base_url = str(request.base_url).rstrip("/")
    return f"{base_url}/go2rtc"


async def _go2rtc_call(method: str, path: str, **kwargs) -> Any:
    if not GO2RTC_API_URL:
        raise RuntimeError("go2rtc is not configured")
    url = f"{GO2RTC_API_URL}{path}"
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.request(method, url, **kwargs)
    if resp.status_code >= 400:
        raise RuntimeError(f"go2rtc {method} {path} failed: HTTP {resp.status_code} {resp.text[:300]}")
    if not resp.content:
        return None
    try:
        return resp.json()
    except Exception:
        return resp.text


async def _go2rtc_register_stream(name: str, rtsp_url: str) -> None:
    try:
        # 1. Register raw stream source
        params_raw = [
            ("name", f"{name}_raw"),
            ("src", rtsp_url)
        ]
        await _go2rtc_call("PUT", "/api/streams", params=params_raw)
        
        # 2. Register always-transcoded compatible H.264 stream for instant browser play
        params_compatible = [
            ("name", name),
            ("src", f"exec:ffmpeg -y -rtsp_transport tcp -i {rtsp_url} -c:v libx264 -preset ultrafast -tune zerolatency -g 25 -an -f mpegts -")
        ]
        await _go2rtc_call("PUT", "/api/streams", params=params_compatible)
    except Exception:
        streams = await _go2rtc_call("GET", "/api/streams")
        if isinstance(streams, dict) and name in streams and f"{name}_raw" in streams:
            return
        raise


async def _go2rtc_unregister_stream(name: str | None) -> None:
    if not name or not GO2RTC_API_URL:
        return
    try:
        await _go2rtc_call("DELETE", "/api/streams", params={"src": name})
    except Exception as exc:
        logger.info("go2rtc unregister failed for %s: %s", name, exc)
    try:
        await _go2rtc_call("DELETE", "/api/streams", params={"src": f"{name}_raw"})
    except Exception as exc:
        logger.info("go2rtc unregister failed for %s_raw: %s", name, exc)


async def _go2rtc_cleanup_stale_streams() -> None:
    """On API startup, remove any triton_* streams left over from a previous
    session (e.g. container crash, forced restart).  go2rtc persists stream
    entries across restarts, so without this the YAML grows without bound."""
    if not GO2RTC_API_URL:
        return
    try:
        streams = await _go2rtc_call("GET", "/api/streams")
    except Exception as exc:
        logger.info("go2rtc not available at startup (cleanup skipped): %s", exc)
        return
    if not isinstance(streams, dict):
        return
    stale = [name for name in streams if name.startswith("triton_")]
    if not stale:
        return
    logger.info("go2rtc: removing %d stale triton stream(s) from previous session: %s", len(stale), stale)
    for name in stale:
        try:
            await _go2rtc_call("DELETE", "/api/streams", params={"src": name})
        except Exception as exc:
            logger.warning("go2rtc cleanup failed for %s: %s", name, exc)


async def _go2rtc_ready() -> bool:
    if not GO2RTC_API_URL:
        return False
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.get(f"{GO2RTC_API_URL}/api/streams")
        resp.raise_for_status()
        return True
    except Exception:
        return False


class ManagedRTSPStream:
    def __init__(self, stream_id: str, cfg: ManagedStreamCreate):
        self.id = stream_id
        self.name = cfg.name or stream_id
        self.url = _validate_rtsp_url(cfg.url)
        self._original_url = cfg.url
        self.backend = _validate_stream_backend(cfg.backend)
        self.requested_models = list(dict.fromkeys(cfg.models or []))
        self.expand_ensembles = bool(getattr(cfg, "expand_ensembles", True))
        self.models = _expand_stream_models(self.requested_models, self.expand_ensembles)
        self.classes = cfg.classes
        self.prompts = cfg.prompts
        self.input_size = parse_imgsz(cfg.imgsz)
        self.conf = float(cfg.conf)
        self.iou = float(cfg.iou)
        self.fps = _clamp_rtsp_fps(float(cfg.fps))
        self.preview_fps = _clamp_rtsp_fps(float(cfg.preview_fps))
        self.max_result_age_ms = int(getattr(cfg, "max_result_age_ms", RTSP_MAX_RESULT_AGE_MS) or RTSP_MAX_RESULT_AGE_MS)
        self.jpeg_quality = _clamp_jpeg_quality(cfg.jpeg_quality)
        self.annotated_preview = bool(cfg.annotated_preview)
        self.source_max_height = _normalize_stream_source_max_height(cfg.source_max_height)
        self.live_transport = (getattr(cfg, "live_transport", None) or "api_jpeg").strip().lower()
        self.tracking_enabled = bool(getattr(cfg, "enable_tracking", False))
        self.recording_enabled = bool(getattr(cfg, "enable_recording", False))
        self.recording_writer = None
        self.recording_process = None
        self.recording_start_time = None
        self.recording_file = None       # path to the .m3u8 playlist
        self.recording_dir = None        # directory holding .ts segments
        self.recording_started_on_disk = False
        self.client_ip: str = "unknown"   # set after construction from request.client.host
        self.go2rtc_name: str | None = None
        self.go2rtc_public_url: str | None = None
        self.go2rtc_error: str | None = None
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.started_at: float | None = None
        self.status = "starting"
        self.error: str | None = None
        self.frame_seq = 0
        self.infer_seq = 0
        self.reconnects = 0
        self.last_event: dict | None = None
        self.last_preview: dict | None = None
        self.last_frame_shape: tuple[int, int] | None = None
        self.last_frame_at: float | None = None
        self.last_infer_at: float | None = None
        self.source_bytes = 0
        self.preview_bytes = 0
        self.event_bytes = 0
        self._events: set[asyncio.Queue] = set()
        self._previews: set[asyncio.Queue] = set()
        self._client_change = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.tracker = BYTETracker(track_thresh=0.45, match_thresh=1.5, max_lost=30)
        self._need_restart = False
        # ByteTrack bbox trail: {local_id: {"pts": [...], "last_ts": float, "point_id": str|None, "cls": str}}
        self._bbox_trail: dict[int, dict] = {}
        self._registered_local_ids: set[int] = set()

    def _stop_recording_process(self) -> None:
        if self.recording_process is not None:
            try:
                logger.info(f"Stopping FFmpeg HLS recording process for stream {self.id}")
                if self.recording_process.stdin:
                    try:
                        self.recording_process.stdin.write(b'q\n')
                        self.recording_process.stdin.flush()
                    except Exception:
                        pass
                try:
                    self.recording_process.wait(timeout=3)
                except Exception:
                    try:
                        self.recording_process.terminate()
                    except Exception:
                        pass
            except Exception as err:
                logger.warning(f"Error while waiting for recording process: {err}")
                try:
                    self.recording_process.terminate()
                except Exception:
                    pass
            self.recording_process = None
            
        if self.recording_file and os.path.exists(self.recording_file):
            try:
                with open(self.recording_file, "r") as f:
                    content = f.read()
                if "#EXT-X-ENDLIST" not in content:
                    with open(self.recording_file, "a") as f:
                        f.write("\n#EXT-X-ENDLIST\n")
                    logger.info(f"Appended #EXT-X-ENDLIST to recording playlist {self.recording_file}")
            except Exception as append_err:
                logger.error(f"Failed to append #EXT-X-ENDLIST to {self.recording_file}: {append_err}")

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name=f"managed-rtsp-{self.id}")

    async def stop(self) -> None:
        if self.status == "stopped":
            return
        self.status = "stopping"
        self._stop.set()
        self._notify_subscribers_closed()
        if self._task:
            try:
                # Do not cancel this task while it is waiting on
                # run_in_executor(_read_rtsp_frame). OpenCV/FFmpeg can still be
                # inside native read(); releasing the VideoCapture concurrently
                # can segfault the whole API process. Let the worker leave its
                # loop and release the capture from its own finally block.
                timeout = max(2.0, (RTSP_READ_TIMEOUT_MS / 1000.0) + 1.0)
                await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(
                    "Managed RTSP stream %s stop is still waiting for camera read to return",
                    self.id,
                )
            except asyncio.CancelledError:
                pass
        self.status = "stopped"
        self._stop_recording_process()

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
            "status": self.status,
            "error": self.error,
            "requested_models": self.requested_models,
            "expand_ensembles": self.expand_ensembles,
            "models": self.models,
            "classes": self.classes,
            "prompts": self.prompts,
            "imgsz": list(self.input_size),
            "conf": self.conf,
            "iou": self.iou,
            "fps": self.fps,
            "preview_fps": self.preview_fps,
            "max_result_age_ms": self.max_result_age_ms,
            "jpeg_quality": self.jpeg_quality,
            "annotated_preview": self.annotated_preview,
            "source_max_height": self.source_max_height,
            "live_transport": self.live_transport,
            "tracking_enabled": self.tracking_enabled,
            "recording_enabled": self.recording_enabled,
            "recording_file": os.path.relpath(self.recording_file, "/app/recordings") if self.recording_file else None,
            "go2rtc_name": self.go2rtc_name,
            "go2rtc_public_url": self.go2rtc_public_url,
            "go2rtc_error": self.go2rtc_error,
            "rtsp_backend": _rtsp_effective_backend(),
            "backend_requested": self.backend,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "frame_seq": self.frame_seq,
            "infer_seq": self.infer_seq,
            "reconnects": self.reconnects,
            "last_frame_at": self.last_frame_at,
            "last_infer_at": self.last_infer_at,
            "last_frame_shape": list(self.last_frame_shape) if self.last_frame_shape else None,
            "event_clients": len(self._events),
            "preview_clients": len(self._previews),
            "source_bytes": self.source_bytes,
            "preview_bytes": self.preview_bytes,
            "event_bytes": self.event_bytes,
        }

    def patch(self, cfg: ManagedStreamPatch) -> dict:
        need_restart = False
        if cfg.backend is not None and cfg.backend != self.backend:
            self.backend = _validate_stream_backend(cfg.backend)
            need_restart = True
        if cfg.models is not None:
            models = list(dict.fromkeys(cfg.models))
            if not models:
                raise HTTPException(400, "models must contain at least one model")
            if models != self.requested_models:
                self.requested_models = models
                need_restart = True
        if cfg.expand_ensembles is not None and bool(cfg.expand_ensembles) != self.expand_ensembles:
            self.expand_ensembles = bool(cfg.expand_ensembles)
            need_restart = True
        if cfg.models is not None or cfg.expand_ensembles is not None:
            self.models = _expand_stream_models(self.requested_models, self.expand_ensembles)
        if cfg.classes is not None and cfg.classes != self.classes:
            self.classes = cfg.classes
            need_restart = True
        if cfg.prompts is not None and cfg.prompts != self.prompts:
            self.prompts = cfg.prompts
            need_restart = True
        if cfg.imgsz is not None:
            new_sz = parse_imgsz(cfg.imgsz)
            if new_sz != self.input_size:
                self.input_size = new_sz
                need_restart = True
        if cfg.conf is not None:
            self.conf = float(cfg.conf)
        if cfg.iou is not None:
            self.iou = float(cfg.iou)
        if cfg.fps is not None:
            self.fps = _clamp_rtsp_fps(float(cfg.fps))
        if cfg.preview_fps is not None:
            self.preview_fps = _clamp_rtsp_fps(float(cfg.preview_fps))
        if cfg.max_result_age_ms is not None:
            self.max_result_age_ms = max(100, int(cfg.max_result_age_ms))
        if cfg.live_transport is not None:
            value = str(cfg.live_transport).strip().lower()
            if value not in {"go2rtc", "api_jpeg"}:
                raise APIError("live_transport must be go2rtc or api_jpeg", 400)
            self.live_transport = value
        if cfg.jpeg_quality is not None:
            self.jpeg_quality = _clamp_jpeg_quality(cfg.jpeg_quality)
        if cfg.annotated_preview is not None:
            self.annotated_preview = bool(cfg.annotated_preview)
        if cfg.source_max_height is not None and cfg.source_max_height != self.source_max_height:
            self.source_max_height = _normalize_stream_source_max_height(cfg.source_max_height)
            need_restart = True
        if cfg.enable_tracking is not None:
            self.tracking_enabled = bool(cfg.enable_tracking)
            if self.tracking_enabled:
                # Reset tracker state when re-enabling
                self.tracker = BYTETracker(track_thresh=0.45, match_thresh=1.5, max_lost=30)
                need_restart = True
        if cfg.enable_recording is not None:
            self.recording_enabled = bool(cfg.enable_recording)
            
        if need_restart:
            self._need_restart = True
            self.tracker = BYTETracker(track_thresh=0.45, match_thresh=1.5, max_lost=30)
            
        self.updated_at = time.time()
        return self.snapshot()

    def subscribe_events(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1)
        self._events.add(q)
        self._client_change.set()
        if self.last_event:
            _queue_latest(q, self.last_event)
        return q

    def unsubscribe_events(self, q: asyncio.Queue) -> None:
        self._events.discard(q)
        self._client_change.set()

    def subscribe_preview(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1)
        self._previews.add(q)
        self._client_change.set()
        if self.last_preview:
            _queue_latest(q, self.last_preview)
        return q

    def unsubscribe_preview(self, q: asyncio.Queue) -> None:
        self._previews.discard(q)
        self._client_change.set()

    def _has_clients(self) -> bool:
        return bool(self._events or self._previews)

    def _notify_subscribers_closed(self) -> None:
        event = {
            "type": "closed",
            "stream_id": self.id,
            "stream_name": self.name,
            "ts": time.time(),
            "stats": self.snapshot(),
        }
        for q in list(self._events):
            _queue_latest(q, event)
        for q in list(self._previews):
            _queue_latest(q, None)

    async def _process_track_reid(self, t, class_name="person", bbox_trail: list = None):
        if not t.crops:
            return
        
        weighted_embs = []
        total_weight = 0.0
        for c in t.crops:
            try:
                frame_for_crop = _decode_jpeg_frame(c["frame_jpeg"])
                emb = await extract_embedding(frame_for_crop, c["bbox"])
                if emb:
                    weight = float(c.get("quality", 1.0))
                    weighted_embs.append(np.array(emb) * weight)
                    total_weight += weight
            except Exception as e:
                logger.warning(f"Keyframe embedding extraction failed: {e}")
        
        if not weighted_embs or total_weight == 0.0:
            return
            
        avg_emb = np.sum(weighted_embs, axis=0) / total_weight
        norm = np.linalg.norm(avg_emb)
        if norm > 0:
            avg_emb = avg_emb / norm
        emb_list = avg_emb.tolist()
        
        gid = None
        existing_point_id = None  # Qdrant point_id for existing global_id (if re-detection)
        if class_name == "person":
            _MIN_TRAVEL_S   = float(os.getenv("MIN_TRAVEL_SECONDS", "5.0"))
            _REID_THRESH    = float(os.getenv("REID_THRESHOLD", "0.85"))
            _MAX_REID_AGE_S = float(os.getenv("MAX_REID_AGE_SECONDS", "4.0"))

            match = await search_object_with_meta(emb_list, class_name=class_name, threshold=_REID_THRESH)
            if match:
                _mcam = match.get("camera_id")
                _mts  = match.get("timestamp")
                _reject = False

                if _mts:
                    try:
                        from datetime import datetime, timezone
                        _mt = datetime.fromisoformat(_mts.replace("Z", "+00:00"))
                        _nt = datetime.now(timezone.utc)
                        _dt_s = (_nt - _mt).total_seconds()

                        if _dt_s > _MAX_REID_AGE_S:
                            _reject = True
                            logger.info(
                                f"[Re-ID AGE REJECT] {match['global_id']} last seen "
                                f"{_dt_s:.1f}s ago (>{_MAX_REID_AGE_S}s) — treating as new entity."
                            )
                        elif _mcam and _mcam != self.id and abs(_dt_s) < _MIN_TRAVEL_S:
                            _reject = True
                            logger.info(
                                f"[Re-ID ST REJECT] {match['global_id']} seen on "
                                f"{_mcam} {abs(_dt_s):.1f}s ago — impossible transit "
                                f"to cam {self.id}. New entity."
                            )
                    except Exception:
                        pass
                if not _reject:
                    gid = match["global_id"]
                    existing_point_id = match.get("point_id")  # Reuse this — no new Qdrant point

        if not gid:
            prefix = class_name[:3].upper() if (class_name and class_name != "unknown") else "OBJ"
            gid = f"{prefix}-{uuid.uuid4().hex[:6].upper()}"

        point_id = None
        if t.crops:
            best_crop = t.crops[0]
            best_frame = _decode_jpeg_frame(best_crop["frame_jpeg"])

            x1, y1, x2, y2 = map(int, best_crop["bbox"])
            img_path = f"/events_images/{gid}_{int(time.time())}.jpg"
            img_path_full = f"/events_images/{gid}_{int(time.time())}_full.jpg"

            if best_frame is not None and best_frame.size > 0:
                crop_img = best_frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if crop_img.size > 0:
                    cv2.imwrite(img_path, crop_img)
                cv2.imwrite(img_path_full, best_frame)

            best_ts = best_crop.get("timestamp", time.time())

            if existing_point_id:
                # Re-detection of known object — reuse existing Qdrant point, NO new point.
                # ONLY update: last_seen timestamp.
                # DO NOT return point_id (prevents caller from overwriting original object's trail).
                # DO NOT update image_path or image_path_full/bbox.
                point_id = None
                now_iso = datetime.utcnow().isoformat() + "Z"
                asyncio.create_task(update_object_last_seen(existing_point_id, now_iso))
            else:
                # Truly new object — create the Qdrant point
                v_fn = os.path.relpath(self.recording_file, "/app/recordings") if self.recording_file else None
                v_off = float(best_ts) - self.recording_start_time if (self.recording_file and self.recording_start_time) else None

                point_id = await add_object_event(
                    gid, emb_list, class_name, self.id, img_path,
                    client_ip=self.client_ip,
                    video_filename=v_fn,
                    video_offset_seconds=v_off,
                    image_path_full=img_path_full,
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    camera_name=self.name,
                    track_session_id=str(t.local_id),
                    bbox_trail=bbox_trail if bbox_trail else None,
                )

        self.tracker.set_global_id(t.local_id, gid)
        t.global_id = gid
        return gid, point_id

    async def _publish_event(self, event: dict) -> None:
        payload_size = len(json.dumps(event, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        self.event_bytes += payload_size * max(1, len(self._events))
        self.last_event = event
        for q in list(self._events):
            _queue_latest(q, event)

    async def _publish_preview(
        self,
        jpeg: bytes,
        frame_seq: int | None = None,
        image_shape: tuple[int, int] | None = None,
    ) -> None:
        self.preview_bytes += len(jpeg) * max(1, len(self._previews))
        item = {
            "jpeg": jpeg,
            "frame_seq": frame_seq,
            "image_shape": list(image_shape) if image_shape else None,
            "ts": time.time(),
        }
        self.last_preview = item
        for q in list(self._previews):
            _queue_latest(q, item)

    async def _infer_models(self, jpeg: bytes, seq: int, capture_ts: float) -> dict:
        started = time.perf_counter()
        # Keep ensemble names when expand_ensembles=false so hybrid/native orchestration runs.
        infer_targets = (
            list(dict.fromkeys(self.requested_models))
            if not self.expand_ensembles
            else self.models
        )
        tasks = [
            _run_inference(
                model=model,
                image_bytes=jpeg,
                image_id=seq,
                conf=self.conf,
                iou=self.iou,
                classes=self.classes,
                prompts=self.prompts,
                input_size=self.input_size,
            )
            for model in infer_targets
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        annotations: list[dict] = []
        per_model: dict[str, Any] = {}
        errors: dict[str, str] = {}
        image_shape = None
        inference_imgsz = list(self.input_size)
        ann_id = 0
        for model, result in zip(infer_targets, results):
            if isinstance(result, Exception):
                errors[model] = str(result)
                continue
            per_model[model] = result.get("timing_ms")
            image_shape = image_shape or result.get("image_shape")
            inference_imgsz = result.get("inference_imgsz", inference_imgsz)
            for ann in result.get("annotations", []):
                ann = dict(ann)
                ann["id"] = ann_id
                ann["source_model"] = ann.get("source_model") or model
                ann_id += 1
                annotations.append(ann)
        total_ms = (time.perf_counter() - started) * 1000
        return {
            "stream_id": self.id,
            "stream_name": self.name,
            "frame_seq": seq,
            "frame_capture_ts": capture_ts,
            "ts": time.time(),
            "models": self.requested_models,
            "infer_models": infer_targets,
            "annotations": annotations,
            "image_shape": image_shape,
            "inference_imgsz": inference_imgsz,
            "timing_ms": {"total": round(total_ms, 3)},
            "per_model_timing_ms": per_model,
            "errors": errors,
            "stats": self.snapshot(),
        }

    async def _run(self) -> None:
        loop = asyncio.get_event_loop()
        cap = None
        self.started_at = time.time()
        try:
            while not self._stop.is_set():
                if not self._has_clients():
                    self.status = "idle"
                    self._client_change.clear()
                    try:
                        await asyncio.wait_for(self._client_change.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        pass
                    continue
                try:
                    self.status = "connecting"
                    cap = await asyncio.wait_for(
                        loop.run_in_executor(None, _open_rtsp_capture, self.url, self.backend),
                        timeout=RTSP_OPEN_TIMEOUT_MS / 1000.0 + 2.0
                    )
                    self.status = "running"
                    self.error = None
                    read_failures = 0
                    last_infer = 0.0
                    last_preview = 0.0
                    infer_tasks: list[tuple[asyncio.Task, bytes]] = []
                    while not self._stop.is_set():
                        if getattr(self, "_need_restart", False):
                            self._need_restart = False
                            logger.info(f"Stream {self.id} settings changed. Restarting capture loop.")
                            for task, _infer_jpeg in infer_tasks:
                                if not task.done():
                                    task.cancel()
                            infer_tasks.clear()
                            try:
                                if cap is not None:
                                    cap.release()
                            except Exception:
                                pass
                            cap = None
                            break

                        if not self._has_clients():
                            for task, _infer_jpeg in infer_tasks:
                                if not task.done():
                                    task.cancel()
                            infer_tasks.clear()
                            self.status = "idle"
                            try:
                                if cap is not None:
                                    cap.release()
                            except Exception:
                                pass
                            cap = None
                            break

                        while infer_tasks and infer_tasks[0][0].done():
                            task, infer_jpeg = infer_tasks.pop(0)
                            try:
                                event = task.result()
                                self.last_infer_at = time.time()
                                result_age_ms = (time.time() - float(event.get("frame_capture_ts", time.time()))) * 1000
                                event["result_age_ms"] = round(result_age_ms, 3)
                                if result_age_ms > self.max_result_age_ms:
                                    logger.info(
                                        "Dropping stale RTSP result stream=%s frame=%s age=%.1fms max=%sms",
                                        self.id,
                                        event.get("frame_seq"),
                                        result_age_ms,
                                        self.max_result_age_ms,
                                    )
                                    continue
                                
                                # --- START MTMC TRACKING LOGIC ---
                                if self.tracking_enabled:
                                  byte_dets = []
                                  annotations = event.get("annotations", [])
                                  for ann in annotations:
                                      x, y, w, h = ann["bbox"]
                                      byte_dets.append({
                                          "bbox": [x, y, x + w, y + h],
                                          "score": float(ann.get("score", 1.0)),
                                      })

                                  track_results = self.tracker.update(byte_dets)

                                  for i, t in enumerate(track_results):
                                      if t is None:
                                          continue
                                      ann = annotations[i]
                                      class_name = ann.get("category_name") or ann.get("label") or ann.get("class") or "object"
                                      
                                      track_obj = self.tracker.get_track(t["local_id"])
                                      if track_obj:
                                          track_obj.add_crop_candidate(
                                              infer_jpeg,
                                              t["bbox"],
                                              float(ann.get("score", 1.0)),
                                              frame_timestamp=event.get("frame_capture_ts")
                                          )

                                      event["annotations"][i]["local_id"] = t["local_id"]
                                      event["annotations"][i]["track_id"] = (
                                          t["global_id"] or f"L{t['local_id']}"
                                      )

                                      # TRAIL & FAST GALLERY: record position nodes when center moves >= 3.0px
                                      # Fast gallery appearance: register event in DB at hits >= 3
                                      lid_t = t["local_id"]
                                      det_score = float(ann.get("score", 1.0))
                                      if lid_t and det_score >= 0.20:
                                          trail_bbox = [float(v) for v in t["bbox"]]
                                          now_ts = event.get("frame_capture_ts") or time.time()
                                          st_id = getattr(track_obj, "track_id", lid_t) if track_obj else lid_t
                                          
                                          is_new_track = (
                                              lid_t not in self._bbox_trail
                                              or self._bbox_trail[lid_t].get("track_id") != st_id
                                          )
                                          if is_new_track:
                                              self._bbox_trail[lid_t] = {
                                                  "track_id": st_id,
                                                  "pts": [trail_bbox],
                                                  "cls": class_name,
                                                  "last_ts": now_ts,
                                                  "point_id": None,
                                              }
                                          entry = self._bbox_trail[lid_t]
                                          pts = entry["pts"]
                                          if not is_new_track and pts:
                                              prev = pts[-1]
                                              # Ground position (bottom-center of bounding box)
                                              prev_cx, prev_feet_y = (prev[0] + prev[2]) / 2, prev[3]
                                              cur_cx, cur_feet_y   = (trail_bbox[0] + trail_bbox[2]) / 2, trail_bbox[3]
                                              moved = ((cur_cx - prev_cx)**2 + (cur_feet_y - prev_feet_y)**2) ** 0.5
                                              dt = now_ts - entry.get("last_ts", 0.0)
                                              if moved > 120.0:
                                                  # Outlier / Teleportation filter: ignore sudden 120px jumps across frame
                                                  pass
                                              elif moved >= 3.0 or dt >= 0.2:
                                                  pts.append(trail_bbox)
                                                  entry["last_ts"] = now_ts
                                                  if len(pts) > 100:
                                                      entry["pts"] = [pts[0]] + pts[-99:]
                                                  if entry.get("point_id"):
                                                      asyncio.create_task(update_object_event_trail(entry["point_id"], entry["pts"]))

                                      # Fast gallery trigger at hit >= 3
                                      if track_obj and track_obj.hits >= 3 and lid_t not in self._registered_local_ids and not t["global_id"]:
                                          self._registered_local_ids.add(lid_t)
                                          try:
                                              gid, pid = await self._process_track_reid(track_obj, class_name, bbox_trail=entry["pts"])
                                              if pid:
                                                  entry["point_id"] = pid
                                              event["annotations"][i]["track_id"] = gid
                                          except Exception as tr_err:
                                              logger.warning(f"Fast Re-ID error for L{lid_t}: {tr_err}")
                                  # --- END BBOX TRAIL ---

                                  if hasattr(self.tracker, "removed") and self.tracker.removed:
                                      for rm_t in list(self.tracker.removed):
                                          lid_rm  = rm_t.local_id
                                          t_entry = self._bbox_trail.pop(lid_rm, None)
                                          self._registered_local_ids.discard(lid_rm)
                                          if t_entry:
                                              t_pts = t_entry["pts"]
                                              pid   = t_entry.get("point_id")
                                              if pid:
                                                  await update_object_event_trail(pid, t_pts)
                                              elif rm_t.crops:
                                                  try:
                                                      await self._process_track_reid(rm_t, t_entry["cls"], bbox_trail=t_pts)
                                                  except Exception as tr_err:
                                                      logger.warning(f"Re-ID error for removed track L{lid_rm}: {tr_err}")
                                      self.tracker.removed = []
                                # --- END MTMC TRACKING LOGIC ---

                                if self.annotated_preview and self._previews:
                                    # Exact-frame mode: draw detections on the same JPEG that
                                    # went to inference. This keeps boxes/masks aligned with the
                                    # displayed frame instead of mixing two different timelines.
                                    infer_frame = _decode_jpeg_frame(infer_jpeg)
                                    preview_frame, preview_scale = _resize_preview_frame(infer_frame)
                                    annotated = _draw_preview_annotations(
                                        preview_frame,
                                        event.get("annotations", []),
                                        preview_scale,
                                    )
                                    annotated_jpeg = _encode_jpeg_frame(annotated, self.jpeg_quality)
                                    await self._publish_preview(
                                        annotated_jpeg,
                                        event.get("frame_seq"),
                                        tuple(annotated.shape[:2]),
                                    )
                                    last_preview = time.monotonic()

                                await self._publish_event(event)
                            except Exception as exc:
                                logger.warning(f"Managed RTSP inference error {self.id}: {exc}")
                                await self._publish_event(
                                    {
                                        "stream_id": self.id,
                                        "stream_name": self.name,
                                        "type": "detections",
                                        "error": str(exc),
                                        "ts": time.time(),
                                        "stats": self.snapshot(),
                                    }
                                )

                        try:
                            frame, shape = await loop.run_in_executor(
                                None, _read_rtsp_frame, cap
                            )
                            if self.source_max_height:
                                frame, _source_scale = _resize_frame_to_limits(
                                frame,
                                    0,
                                    self.source_max_height,
                                )
                                shape = frame.shape[:2]
                            read_failures = 0
                        except Exception as exc:
                            read_failures += 1
                            if read_failures >= RTSP_MAX_READ_FAILURES:
                                raise exc
                            await asyncio.sleep(0.02)
                            continue

                        now = time.monotonic()
                        self.frame_seq += 1
                        seq = self.frame_seq
                        self.last_frame_shape = shape
                        self.last_frame_at = time.time()

                        # Check if HLS recording has written its first segment on disk, and adjust recording_start_time
                        if self.recording_process is not None and not self.recording_started_on_disk:
                            seg0 = os.path.join(self.recording_dir, "seg_00000.ts") if self.recording_dir else None
                            if seg0 and os.path.exists(seg0):
                                try:
                                    seg_dur = 10.0  # default fallback
                                    playlist_path = self.recording_file
                                    pdt_synced = False
                                    if playlist_path and os.path.exists(playlist_path):
                                        with open(playlist_path, "r") as f:
                                            content = f.read()
                                        import re
                                        # First check for absolute program date time tag
                                        pdt_match = re.search(r"#EXT-X-PROGRAM-DATE-TIME:([^\n]+)", content)
                                        if pdt_match:
                                            try:
                                                pdt_str = pdt_match.group(1).strip()
                                                if pdt_str.endswith('Z'):
                                                    pdt_str = pdt_str[:-1] + '+00:00'
                                                from datetime import datetime
                                                dt = datetime.fromisoformat(pdt_str)
                                                self.recording_start_time = dt.timestamp()
                                                pdt_synced = True
                                                logger.info(f"HLS recording start time synchronized via program-date-time tag: {self.recording_start_time}")
                                            except Exception as parse_err:
                                                logger.warning(f"Failed parsing PROGRAM-DATE-TIME '{pdt_str}': {parse_err}")
                                        
                                        # Match #EXTINF:10.080000,\nseg_00000.ts or #EXTINF:10.080000,seg_00000.ts
                                        match = re.search(r"#EXTINF:([0-9.]+),\s*(?:.*/)?seg_00000\.ts", content)
                                        if match:
                                            seg_dur = float(match.group(1))
                                            logger.info(f"HLS segment duration parsed: {seg_dur}s for seg_00000.ts")
                                    
                                    if not pdt_synced:
                                        mtime = os.path.getmtime(seg0)
                                        self.recording_start_time = mtime - seg_dur
                                        logger.info(f"HLS recording start time adjusted via file mtime: {self.recording_start_time}")
                                        
                                    self.recording_started_on_disk = True
                                except Exception as err:
                                    logger.warning(f"Failed to inspect seg_00000.ts: {err}")

                        if self.recording_enabled:
                            if self.recording_process is None:
                                try:
                                    # Frigate-style: HLS segmented recording
                                    # Each segment is 10s .ts; m3u8 playlist grows as segments are written
                                    # Browser can play the m3u8 live while recording continues
                                    self.recording_start_time = time.time()
                                    self.recording_dir = f"/app/recordings/{self.id}_{int(self.recording_start_time)}"
                                    os.makedirs(self.recording_dir, exist_ok=True)
                                    self.recording_file = os.path.join(self.recording_dir, "live.m3u8")
                                    self.recording_started_on_disk = False


                                    rec_url = getattr(self, "_original_url", self.url)
                                    logger.info(f"Starting HLS segment recording for stream {self.id} → {self.recording_dir}")
                                    cmd = [
                                        "ffmpeg",
                                        "-y",
                                        "-rtsp_transport", "tcp",
                                        "-i", rec_url,
                                        "-c", "copy",
                                        "-reset_timestamps", "1",
                                        "-f", "hls",
                                        "-hls_time", "10",
                                        "-hls_list_size", "0",         # keep ALL segments in playlist
                                        "-hls_flags", "append_list+program_date_time",   # append new segments, write program-date-time
                                        "-hls_segment_type", "mpegts",
                                        "-hls_segment_filename", os.path.join(self.recording_dir, "seg_%05d.ts"),
                                        self.recording_file,
                                    ]
                                    self.recording_process = subprocess.Popen(
                                        cmd,
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL
                                    )
                                except Exception as rec_err:
                                    logger.error(f"Failed to start HLS recording for stream {self.id}: {rec_err}")
                                    self.recording_process = None
                                    self.recording_file = None
                                    self.recording_dir = None
                        else:
                                self._stop_recording_process()
                                # Do NOT clear self.recording_file here — the .ts segments and
                                # .m3u8 playlist still exist on disk and the frontend should
                                # be able to play them after recording stops.


                        preview_due = self.preview_fps > 0 and now - last_preview >= 1.0 / self.preview_fps
                        infer_has_output = bool(self._events or (self.annotated_preview and self._previews))
                        infer_due = (
                            infer_has_output
                            and self.fps > 0
                            and now - last_infer >= 1.0 / self.fps
                        )

                        raw_jpeg: bytes | None = None
                        if preview_due and self._previews and not self.annotated_preview:
                            preview_frame, _preview_scale = _resize_preview_frame(frame)
                            preview_jpeg = _encode_jpeg_frame(preview_frame, self.jpeg_quality)
                            if preview_frame is frame:
                                raw_jpeg = preview_jpeg
                            await self._publish_preview(preview_jpeg, seq, tuple(preview_frame.shape[:2]))
                            last_preview = now

                        if (
                            infer_due
                            and self.models
                            and len(infer_tasks) < RTSP_MAX_INFLIGHT_INFER
                        ):
                            if raw_jpeg is None:
                                infer_frame, _infer_scale = _resize_infer_frame(frame)
                                raw_jpeg = _encode_jpeg_frame(infer_frame, self.jpeg_quality)
                            self.source_bytes += len(raw_jpeg)
                            last_infer = now
                            self.infer_seq = seq
                            infer_tasks.append((asyncio.create_task(self._infer_models(raw_jpeg, seq, self.last_frame_at or time.time())), raw_jpeg))

                        # Yield to the event loop, but only until the next
                        # inference or preview frame is due.  A bare sleep(0)
                        # spins at the camera's native frame-rate (25-30 Hz)
                        # and wastes CPU across 30-40 concurrent camera tasks.
                        _LOOP_HEADROOM_S = 0.002  # 2 ms wakeup headroom
                        _LOOP_MIN_SLEEP_S = 0.001  # at least 1 ms to let I/O run
                        _now = time.monotonic()
                        _candidates: list[float] = []
                        if self.fps > 0:
                            _candidates.append(1.0 / self.fps - (_now - last_infer))
                        if self.preview_fps > 0:
                            _candidates.append(1.0 / self.preview_fps - (_now - last_preview))
                        if _candidates:
                            _sleep = max(_LOOP_MIN_SLEEP_S, min(_candidates) - _LOOP_HEADROOM_S)
                        else:
                            _sleep = 0.005  # 5 ms fallback when both FPS are 0 (unlimited)
                        await asyncio.sleep(_sleep)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.status = "error"
                    self.error = str(exc)
                    self.reconnects += 1
                    logger.warning(f"Managed RTSP stream {self.id} error: {exc}")
                    try:
                        if cap is not None:
                            cap.release()
                    except Exception:
                        pass
                    cap = None
                    await asyncio.sleep(min(5.0, 0.5 + self.reconnects * 0.25))
        finally:
            pending_infer = locals().get("infer_task")
            if pending_infer:
                pending_task = pending_infer[0] if isinstance(pending_infer, tuple) else pending_infer
                if not pending_task.done():
                    pending_task.cancel()
            self._stop_recording_process()

            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            if self.status != "stopped":
                self.status = "stopped"


managed_streams: dict[str, ManagedRTSPStream] = {}


async def _drain_model_from_streams(model_name: str, wait_s: float = 0.12) -> None:
    """
    Temporarily remove *model_name* from all active streams so that no new
    inference tasks are scheduled on it.  After ``wait_s`` seconds any
    in-flight gRPC calls will have completed and it is safe to unload.
    Call ``_restore_model_to_streams`` after the reload.

    Returns a list of stream IDs that were affected (pass to restore).
    """
    # Exposed as a module-level coroutine so config_put / instances_put can use it
    # without importing anything extra.
    affected = []
    for sid, stream in list(managed_streams.items()):
        if model_name in stream.models:
            stream.models = [m for m in stream.models if m != model_name]
            affected.append(sid)
    if affected:
        # One inference cycle is typically 30-100 ms; 120 ms is conservative.
        await asyncio.sleep(wait_s)
    return affected  # type: ignore[return-value]


def _restore_model_to_streams(model_name: str, affected_ids: list[str]) -> None:
    """Re-add *model_name* to the streams that were drained before reload."""
    for sid in affected_ids:
        stream = managed_streams.get(sid)
        if stream is None:
            continue
        # Re-expand from requested_models to get the canonical ordered list
        expanded = _expand_stream_models(stream.requested_models, stream.expand_ensembles)
        stream.models = expanded



class RTSPProbeRequest(BaseModel):
    url: str
    jpeg_quality: int = RTSP_JPEG_QUALITY


class ManagedStreamCreate(BaseModel):
    name: str | None = None
    url: str
    models: list[str]
    backend: str | None = Field(
        None,
        description="Optional stream backend override: auto, gstreamer, or opencv.",
    )
    expand_ensembles: bool = True
    classes: str | None = None
    prompts: str | None = None
    imgsz: str | int | None = None
    conf: float = DEFAULT_CONF
    iou: float = DEFAULT_IOU
    fps: float = 10.0
    preview_fps: float = 5.0
    max_result_age_ms: int = Field(
        RTSP_MAX_RESULT_AGE_MS,
        description="Drop detection results older than this many milliseconds instead of drawing stale boxes.",
    )
    live_transport: str = Field(
        "go2rtc",
        description="RTSP live-view transport: go2rtc for WebRTC native preview, or api_jpeg for API preview WebSocket.",
    )
    source_max_height: int | None = Field(
        None,
        description="Optional RTSP working resolution cap by height: native/0, 512, 640, 720, 960, 1080, or 1440.",
    )
    jpeg_quality: int = RTSP_JPEG_QUALITY
    annotated_preview: bool = Field(
        True,
        description=(
            "If true, preview WebSocket sends exact inference-frame JPEGs with boxes drawn on the API server. "
            "If false, preview sends raw camera frames; clients draw from JSON on /events."
        ),
    )
    enable_tracking: bool = Field(
        False,
        description="Enable ByteTrack object tracking + Re-ID on this stream. Disabled by default.",
    )
    enable_recording: bool = Field(
        False,
        description="Enable clean server-side video recording to MP4.",
    )


class ManagedStreamPatch(BaseModel):
    models: list[str] | None = None
    backend: str | None = Field(
        None,
        description="Optional stream backend override: auto, gstreamer, or opencv.",
    )
    expand_ensembles: bool | None = None
    classes: str | None = None
    prompts: str | None = None
    imgsz: str | int | None = None
    conf: float | None = None
    iou: float | None = None
    fps: float | None = None
    enable_tracking: bool | None = None
    enable_recording: bool | None = None
    preview_fps: float | None = None
    max_result_age_ms: int | None = Field(
        None,
        description="Drop detection results older than this many milliseconds instead of drawing stale boxes.",
    )
    live_transport: str | None = Field(
        None,
        description="RTSP live-view transport: go2rtc or api_jpeg.",
    )
    source_max_height: int | None = Field(
        None,
        description="Optional RTSP working resolution cap by height: native/0, 512, 640, 720, 960, 1080, or 1440.",
    )
    jpeg_quality: int | None = None
    annotated_preview: bool | None = Field(
        None,
        description="Toggle exact-frame server-drawn preview boxes. false = raw preview JPEG + client overlay.",
    )


def _expand_stream_models(models: list[str], expand_ensembles: bool = True) -> list[str]:
    """
    Live RTSP favors low latency. Expanding ensembles lets `/streams` run the
    sub-models as parallel API jobs, which matches selecting the single models
    directly in the web client.
    """
    if not expand_ensembles:
        return list(dict.fromkeys(models))

    expanded: list[str] = []
    seen: set[str] = set()

    def add_model(name: str, depth: int = 0) -> None:
        if name in seen or depth > 4:
            return
        if is_ensemble(MODEL_REPO_PATH, name):
            try:
                for step in parse_ensemble_steps(MODEL_REPO_PATH, name):
                    sub = step.get("model")
                    if sub:
                        add_model(sub, depth + 1)
                return
            except Exception:
                pass
        seen.add(name)
        expanded.append(name)

    for model in models:
        add_model(model)
    return expanded


def _encoder_for_model(model_name: str) -> YOLOETextEncoder:
    """Global YOLOE_WEIGHTS or per-model encoder.pt saved at upload."""
    per_model = os.path.join(MODEL_REPO_PATH, model_name, "encoder.pt")
    if os.path.exists(per_model) and model_name not in _bad_model_encoders:
        try:
            if model_name not in _encoders_by_model:
                _encoders_by_model[model_name] = YOLOETextEncoder(per_model)
            return _encoders_by_model[model_name]
        except Exception as exc:
            _encoders_by_model.pop(model_name, None)
            _bad_model_encoders.add(model_name)
            logger.warning(
                "Could not load YOLOE encoder for %s from %s; falling back to global encoder: %s",
                model_name,
                per_model,
                exc,
            )
    if encoder is not None:
        return encoder
    raise HTTPException(
        503,
        f"YOLOE encoder not available for '{model_name}'. Mount weights at YOLOE_WEIGHTS "
        "or upload a valid .pt so encoder.pt is stored with the model.",
    )


async def _encode_prompts_for_model(
    model_name: str,
    class_tuple: tuple[str, ...],
    loop: asyncio.AbstractEventLoop,
) -> np.ndarray:
    """Encode YOLOE prompts with per-model encoder, then fallback globally once."""
    enc = _encoder_for_model(model_name)
    try:
        return await loop.run_in_executor(None, enc.encode, class_tuple)
    except Exception as exc:
        per_model = os.path.join(MODEL_REPO_PATH, model_name, "encoder.pt")
        if os.path.exists(per_model) and model_name not in _bad_model_encoders:
            _encoders_by_model.pop(model_name, None)
            _bad_model_encoders.add(model_name)
            logger.warning(
                "YOLOE encoder encode failed for %s from %s; retrying with global encoder: %s",
                model_name,
                per_model,
                exc,
            )
            if encoder is not None:
                return await loop.run_in_executor(None, encoder.encode, class_tuple)
        raise


async def _infer_submodel(
    submodel: str,
    tensor: np.ndarray,
    meta: dict,
    image_id: int,
    conf: float,
    iou: float,
    classes: Optional[str],
    prompts: Optional[str] = None,
    step_info: Optional[dict] = None,
    timing_bucket: Optional[dict[str, float]] = None,
) -> dict:
    """Run one sub-model (used by hybrid ensembles)."""
    loop = asyncio.get_event_loop()
    if step_info and step_info.get("model_type"):
        sub_info = {
            "type": step_info["model_type"],
            "has_masks": step_info.get("has_masks", False),
            "output0_layout": step_info.get("output0_layout"),
        }
    else:
        sub_info = _model_info(submodel)

    prompt_embedding: Optional[np.ndarray] = None
    class_names: Optional[list[str]] = None

    if sub_info["type"] == MODEL_TYPE_YOLOE_DYNAMIC:
        prompt_list = parse_text_prompts(prompts, classes)
        if not prompt_list:
            raise HTTPException(
                400,
                f"YOLOE step '{submodel}' requires 'prompts' "
                "(comma-separated text, e.g. person,car)",
            )
        class_tuple = tuple(prompt_list)
        t0 = time.perf_counter()
        prompt_embedding = await _encode_prompts_for_model(submodel, class_tuple, loop)
        if timing_bucket is not None:
            timing_bucket["encode"] = timing_bucket.get("encode", 0.0) + (
                time.perf_counter() - t0
            ) * 1000
        class_names = list(class_tuple)
    else:
        # Standard YOLO: labels.json only; optional classes overrides labels.
        # prompts is ignored — never bleeds YOLOE text into YOLO naming.
        if classes:
            class_names = [c.strip() for c in classes.split(",") if c.strip()]
        else:
            class_names = read_labels(MODEL_REPO_PATH, submodel)

    paused = _model_disabled_message(submodel)
    if paused:
        raise HTTPException(503, paused)

    async with semaphore:
        t0 = time.perf_counter()
        try:
            outputs = await loop.run_in_executor(
                None, triton.infer, submodel, tensor, prompt_embedding
            )
            _record_model_success(submodel)
        except Exception as exc:
            _record_model_failure(submodel, exc)
            raise
    if timing_bucket is not None:
        timing_bucket["triton"] = timing_bucket.get("triton", 0.0) + (
            time.perf_counter() - t0
        ) * 1000

    t0 = time.perf_counter()
    result = postprocess(
        outputs=outputs,
        meta=meta,
        conf_threshold=conf,
        iou_threshold=iou,
        class_names=class_names,
        image_id=image_id,
        has_masks=sub_info.get("has_masks", False),
        output_layout=sub_info.get("output0_layout"),
    )
    if timing_bucket is not None:
        timing_bucket["postprocess"] = timing_bucket.get("postprocess", 0.0) + (
            time.perf_counter() - t0
        ) * 1000
    return result


async def _run_hybrid_ensemble(
    steps: list[dict],
    tensor: np.ndarray,
    meta: dict,
    image_id: int,
    conf: float,
    iou: float,
    classes: Optional[str],
    prompts: Optional[str] = None,
    timing_bucket: Optional[dict[str, float]] = None,
    sequential: bool = False,
    allow_partial: bool = True,
) -> dict:
    """API-orchestrated ensemble for mixed YOLO + YOLOE steps."""
    async def _run_step(step: dict) -> dict:
        return await _infer_submodel(
            step["model"],
            tensor,
            meta,
            image_id,
            conf,
            iou,
            classes,
            prompts,
            step,
            timing_bucket,
        )

    parts: list[dict | BaseException] = []
    if sequential:
        for step in steps:
            try:
                parts.append(await _run_step(step))
            except Exception as exc:
                if not allow_partial:
                    raise
                logger.warning(
                    "Ensemble step failed and was skipped: %s: %s",
                    step.get("model"),
                    exc,
                )
                parts.append(exc)
    else:
        parts = list(await asyncio.gather(*[_run_step(step) for step in steps], return_exceptions=allow_partial))

    orig_h, orig_w = meta["orig_shape"]
    merged: list[dict] = []
    errors: list[dict[str, str]] = []
    ann_id = 0
    for step, part in zip(steps, parts):
        if isinstance(part, BaseException):
            errors.append({"model": str(step.get("model", "?")), "error": str(part)})
            continue
        for ann in part["annotations"]:
            ann = dict(ann)
            ann["id"] = ann_id
            ann["source_model"] = step["model"]
            ann_id += 1
            merged.append(ann)

    return {
        "image_id": image_id,
        "image_shape": [orig_h, orig_w],
        "ensemble": True,
        "ensemble_kind": "hybrid",
        "annotations": merged,
        **({"ensemble_errors": errors} if errors else {}),
    }


async def _run_inference(
    model: str,
    image_bytes: bytes,
    image_id: int,
    conf: float,
    iou: float,
    classes: Optional[str] = None,
    prompts: Optional[str] = None,
    input_size: tuple[int, int] = (DEFAULT_IMGSZ, DEFAULT_IMGSZ),
) -> dict:
    """Shared inference + postprocess for /detect and WebSocket stream."""
    timing = {
        "preprocess": 0.0,
        "encode": 0.0,
        "triton": 0.0,
        "postprocess": 0.0,
    }
    total_t0 = time.perf_counter()
    t0 = time.perf_counter()
    tensor, meta = preprocess(image_bytes, input_size=input_size)
    timing["preprocess"] += (time.perf_counter() - t0) * 1000
    info = _model_info(model)
    loop = asyncio.get_event_loop()

    if info["type"] == MODEL_TYPE_ENSEMBLE:
        if info.get("ensemble_kind") == "hybrid":
            result = await _run_hybrid_ensemble(
                info["steps"], tensor, meta, image_id, conf, iou, classes, prompts, timing
            )
            timing["total"] = (time.perf_counter() - total_t0) * 1000
            result["timing_ms"] = _round_timing(timing)
            return result

        try:
            paused = _model_disabled_message(model)
            if paused:
                raise RuntimeError(paused)
            async with semaphore:
                t0 = time.perf_counter()
                _, named = await loop.run_in_executor(
                    None,
                    lambda: triton.infer(model, tensor, None, "1", True),
                )
                _record_model_success(model)
            timing["triton"] += (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            out = postprocess_ensemble(
                named_outputs=named,
                steps=info["steps"],
                meta=meta,
                model_repo=MODEL_REPO_PATH,
                conf_threshold=conf,
                iou_threshold=iou,
                image_id=image_id,
            )
            timing["postprocess"] += (time.perf_counter() - t0) * 1000
            out["ensemble_kind"] = "native"
            timing["total"] = (time.perf_counter() - total_t0) * 1000
            out["timing_ms"] = _round_timing(timing)
            return out
        except Exception as exc:
            _record_model_failure(model, exc)
            logger.warning(
                "Native ensemble %s failed; falling back to sequential API orchestration: %s",
                model,
                exc,
            )
            try:
                await triton_unload_model(TRITON_HTTP_URL, model)
                await asyncio.sleep(0.1)
            except Exception as unload_exc:
                logger.warning("Could not unload failed native ensemble %s before fallback: %s", model, unload_exc)
            result = await _run_hybrid_ensemble(
                info["steps"],
                tensor,
                meta,
                image_id,
                conf,
                iou,
                classes,
                prompts,
                timing,
                sequential=True,
                allow_partial=True,
            )
            result["ensemble_kind"] = "native_fallback"
            result["native_error"] = str(exc)
            timing["total"] = (time.perf_counter() - total_t0) * 1000
            result["timing_ms"] = _round_timing(timing)
            return result

    prompt_embedding: Optional[np.ndarray] = None
    class_names: Optional[list[str]] = None

    if info["type"] == MODEL_TYPE_YOLOE_DYNAMIC:
        prompt_list = parse_text_prompts(prompts, classes)
        if not prompt_list:
            raise HTTPException(
                400,
                "YOLOE dynamic models require 'prompts' "
                "(comma-separated text, e.g. person,car). "
                "Use prompts — not classes — so YOLO labels are not affected.",
            )
        class_tuple = tuple(prompt_list)
        t0 = time.perf_counter()
        prompt_embedding = await _encode_prompts_for_model(model, class_tuple, loop)
        timing["encode"] += (time.perf_counter() - t0) * 1000
        class_names = list(class_tuple)
    else:
        if classes:
            class_names = [c.strip() for c in classes.split(",") if c.strip()]
        else:
            class_names = read_labels(MODEL_REPO_PATH, model)

    paused = _model_disabled_message(model)
    if paused:
        raise HTTPException(503, paused)

    async with semaphore:
        t0 = time.perf_counter()
        try:
            outputs = await loop.run_in_executor(
                None, triton.infer, model, tensor, prompt_embedding
            )
            _record_model_success(model)
        except Exception as exc:
            _record_model_failure(model, exc)
            raise
    timing["triton"] += (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    result = postprocess(
        outputs=outputs,
        meta=meta,
        conf_threshold=conf,
        iou_threshold=iou,
        class_names=class_names,
        image_id=image_id,
        has_masks=info.get("has_masks", False),
        output_layout=info.get("output0_layout"),
    )
    timing["postprocess"] += (time.perf_counter() - t0) * 1000
    timing["total"] = (time.perf_counter() - total_t0) * 1000
    result["timing_ms"] = _round_timing(timing)
    return result


# ─────────────────────────── lifespan ────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global encoder, triton, semaphore

    logger.info("=== API server starting ===")

    try:
        from yoloe_assets import ensure_mobileclip_asset

        ensure_mobileclip_asset()
    except Exception as exc:
        logger.warning("MobileCLIP asset check failed: %s", exc)

    triton = TritonGRPCClient(TRITON_GRPC_URL)

    if os.path.exists(YOLOE_WEIGHTS):
        try:
            encoder = YOLOETextEncoder(YOLOE_WEIGHTS)
        except Exception as exc:
            logger.warning(f"YOLOE encoder failed to load: {exc}")
    else:
        logger.warning(
            f"YOLOE weights not found at {YOLOE_WEIGHTS}. "
            "YOLOE dynamic models will be unavailable."
        )

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    await _sync_triton_models_from_disk()
    await _go2rtc_cleanup_stale_streams()
    logger.info(f"Semaphore: MAX_CONCURRENT={MAX_CONCURRENT}  MAX_FPS={MAX_FPS}")
    logger.info("=== API server ready ===")
    yield
    logger.info("=== API server shutting down ===")
    for stream in list(managed_streams.values()):
        try:
            await stream.stop()
            await _go2rtc_unregister_stream(stream.go2rtc_name)
        except Exception as exc:
            logger.warning(f"Failed to stop managed stream {stream.id}: {exc}")


_OPENAPI_DESCRIPTION = """
Vision inference API (port **8003**). Clients send images or streams; the API preprocesses,
calls **NVIDIA Triton** over gRPC, and returns **COCO-style** JSON (boxes + optional masks).

## Model types
| Type | Labels | YOLOE text |
|------|--------|------------|
| Standard YOLO | `labels.json` or optional `classes` override | — |
| YOLOE dynamic | — | **`prompts`** required (not `classes`) |
| Hybrid ensemble | YOLO steps use `labels.json` | YOLOE steps use **`prompts`** only |

## Docs
- Usage & API reference: `USAGE.md`
- System Architecture: `ARCHITECTURE.md`
- Project Overview: `README.md`

## Interactive UI
- Swagger: `/docs`
- ReDoc: `/redoc`
- OpenAPI JSON: `/openapi.json`
"""

_OPENAPI_TAGS = [
    {
        "name": "System",
        "description": "Health, GPU discovery, host/Triton resource status.",
    },
    {
        "name": "Inference",
        "description": "Single-image `/detect` and live `/ws/stream` (JPEG frames).",
    },
    {
        "name": "Streams",
        "description": "Production RTSP streams owned by the API server.",
    },
    {
        "name": "Models",
        "description": "Upload `.pt` → ONNX + Triton load, list, delete.",
    },
    {
        "name": "Labels",
        "description": "`labels.json` for standard YOLO class names (not used for YOLOE prompts).",
    },
    {
        "name": "Config",
        "description": "Read/update `config.pbtxt` and `instance_group` (hot-reload).",
    },
    {
        "name": "Ensemble",
        "description": "Create, inspect, update, and delete ensemble pipelines.",
    },
]

app = FastAPI(
    title="Vision Inference API",
    version="1.2.0",
    description=_OPENAPI_DESCRIPTION,
    lifespan=lifespan,
    openapi_tags=_OPENAPI_TAGS,
)

async def _periodic_retention_loop():
    while True:
        try:
            await purge_expired_data()
        except Exception as e:
            print(f"[Retention Task] Error: {e}")
        # Run retention cleanup every 6 hours
        await asyncio.sleep(21600)

async def _restore_persistent_streams():
    """
    On API startup, restore all persistent streams from SQLite persistent_streams table.
    RTSP streams run continuously in background as an NVR engine.
    Webcam and video file sources are restored into UI registry but remain paused until requested.
    """
    try:
        from auth import db_fetch_all
        rows = await db_fetch_all("SELECT * FROM persistent_streams")
        if not rows:
            return
        logger.info("[NVR Persistent] Restoring %d stream(s) from persistent storage...", len(rows))
        for r in rows:
            sid = r["stream_id"]
            url = r["url"]
            source_type = r["source_type"] or ("rtsp" if url.startswith(("rtsp://", "rtsps://", "http://", "https://")) else "webcam")
            models = [m.strip() for m in (r["models"] or "").split(",") if m.strip()]
            
            req_data = {
                "name": r["name"] or sid,
                "url": url,
                "models": models,
                "classes": r["classes"],
                "prompts": r["prompts"],
                "imgsz": r["imgsz"] or 640,
                "conf": r["conf"] if r["conf"] is not None else 0.5,
                "fps": r["fps"] or 30,
                "preview_fps": r["preview_fps"] or 10,
                "source_max_height": r["source_max_height"] or 720,
                "enable_tracking": bool(r["enable_tracking"]),
                "enable_recording": bool(r["enable_recording"]),
                "live_transport": r["live_transport"] or "go2rtc",
            }
            req = ManagedStreamCreate(**req_data)
            stream = ManagedRTSPStream(sid, req)
            stream._original_url = url
            stream.client_ip = r["client_ip"] or "system"
            stream.live_transport = r["live_transport"] or "go2rtc"

            if _go2rtc_available():
                stream.go2rtc_name = _go2rtc_stream_name(sid)
                try:
                    await _go2rtc_register_stream(stream.go2rtc_name, url)
                    stream.url = f"rtsp://go2rtc:8554/{stream.go2rtc_name}_raw"
                except Exception as exc:
                    logger.warning("NVR Restore: go2rtc register error for %s: %s", sid, exc)

            managed_streams[sid] = stream
            
            # RTSP streams run continuously in background (NVR mode)
            if source_type == "rtsp" or url.startswith(("rtsp://", "rtsps://", "http://", "https://")):
                logger.info("[NVR Persistent] Auto-starting background RTSP NVR stream: %s (%s)", r["name"], sid)
                stream.start()
                if bool(r["enable_recording"]):
                    try:
                        await stream.start_recording(format=r["rec_format"] or "hls")
                    except Exception as e:
                        logger.error("NVR Restore: recording start error for %s: %s", sid, e)
            else:
                logger.info("[NVR Persistent] Registered non-RTSP stream %s (%s) for UI restoration", r["name"], sid)

    except Exception as e:
        logger.error("[NVR Persistent] Error restoring streams: %s", e, exc_info=True)


@app.on_event("startup")
async def startup_event():
    await init_db()
    asyncio.create_task(_periodic_retention_loop())
    asyncio.create_task(_restore_persistent_streams())

@app.on_event("shutdown")
async def shutdown_event():
    pass

from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# 1. CORS Hardening
_cors_origins_env = os.getenv("CORS_ORIGINS", "")
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
if not _cors_origins or "*" in _cors_origins:
    _cors_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["*"],
)

# 2. Trusted Host Middleware (Host Header Injection defense)
_allowed_hosts_env = os.getenv("ALLOWED_HOSTS", "")
_allowed_hosts = [h.strip() for h in _allowed_hosts_env.split(",") if h.strip()]
if not _allowed_hosts or "*" in _allowed_hosts:
    _allowed_hosts = ["*"]

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=_allowed_hosts,
)

# 3. Custom security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://cdn.tailwindcss.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net; "
        "font-src 'self' https://fonts.gstatic.com; "
        "connect-src 'self' ws: wss: http: https:; "
        "media-src 'self' blob:; "
        "img-src 'self' data: blob:;"
    )
    path = request.url.path
    # Prevent Chrome from caching API responses or admin JS/HTML across sessions.
    # This is the root cause of stale model/ensemble lists showing after another session
    # deletes them — Chrome serves the old cached GET response instead of hitting the server.
    if path.startswith("/admin") or not path.startswith("/static"):
        ct = response.headers.get("content-type", "")
        # Skip binary media (images, video, recordings) — only target JSON/HTML/JS/CSS
        if not any(ct.startswith(m) for m in ("image/", "video/", "audio/")):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
    return response


# 4. Authentication & API Key Management Endpoints
from pydantic import BaseModel
from fastapi import Depends, status
from auth import (
    authenticate_admin, create_admin_session, delete_admin_session,
    validate_admin_session, require_admin, generate_api_key,
    revoke_api_key, update_api_key, list_active_api_keys, verify_api_key,
    authenticate_websocket, db_execute, db_fetch_one, db_fetch_all,
    get_admin_username_from_session,
)

class LoginRequest(BaseModel):
    username: str
    password: str
    remember: Optional[bool] = False

class CreateKeyRequest(BaseModel):
    name: str
    expires_in_days: Optional[int] = None
    scopes: list[str] = ["inference"]
    allowed_models: list[str] = ["*"]

@app.post("/api/v1/auth/login", tags=["Authentication"])
async def login_endpoint(payload: LoginRequest):
    is_valid = await authenticate_admin(payload.username, payload.password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tên tài khoản hoặc mật khẩu không chính xác."
        )
    session_id = await create_admin_session(payload.username)
    response = JSONResponse(content={"status": "ok", "message": "Đăng nhập thành công"})
    
    # 30 days max_age for remember me cookies
    max_age = 30 * 24 * 3600 if payload.remember else None
    
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        samesite="lax",
        path="/",
        max_age=max_age
    )
    return response

@app.post("/api/v1/auth/logout", tags=["Authentication"])
async def logout_endpoint(request: Request):
    session_id = request.cookies.get("session_id")
    if session_id:
        await delete_admin_session(session_id)
    response = JSONResponse(content={"status": "ok", "message": "Đăng xuất thành công"})
    response.delete_cookie(key="session_id", path="/")
    return response

@app.get("/api/v1/auth/status", tags=["Authentication"])
async def auth_status_endpoint(request: Request):
    session_id = request.cookies.get("session_id")
    is_logged_in = await validate_admin_session(session_id)
    return {"logged_in": is_logged_in}

@app.post("/api/v1/admin/keys", tags=["Admin Key Management"])
async def create_key_endpoint(payload: CreateKeyRequest, admin_session: str = Depends(require_admin)):
    valid_scopes = {"admin", "inference", "data:read"}
    for s in payload.scopes:
        if s not in valid_scopes:
            raise HTTPException(status_code=400, detail=f"Quyền không hợp lệ: {s}")
            
    # Get the admin username from the current session for audit tracking
    creator_username = await get_admin_username_from_session(admin_session)
    raw_key, key_id = await generate_api_key(
        name=payload.name,
        expires_in_days=payload.expires_in_days,
        scopes=payload.scopes,
        allowed_models=payload.allowed_models,
        created_by=creator_username,
    )
    return {
        "status": "ok",
        "key_id": key_id,
        "api_key": raw_key,
        "warning": "Vui lòng lưu khóa này ở nơi an toàn. Bạn sẽ không thể nhìn thấy lại khóa này."
    }

@app.get("/api/v1/admin/keys", tags=["Admin Key Management"])
async def list_keys_endpoint(admin_session: str = Depends(require_admin)):
    keys = await list_active_api_keys()
    return {"keys": keys}

class UpdateKeyRequest(BaseModel):
    name: Optional[str] = None
    expires_at: Optional[int] = None

@app.put("/api/v1/admin/keys/{key_id}", tags=["Admin Key Management"])
async def update_key_endpoint(key_id: int, req: UpdateKeyRequest, admin_session: str = Depends(require_admin)):
    success = await update_api_key(key_id, key_name=req.name, expires_at=req.expires_at)
    if not success:
        raise HTTPException(400, "Không có trường hợp lệ để cập nhật.")
    return {"status": "ok", "message": f"API Key {key_id} đã được cập nhật thành công."}

@app.delete("/api/v1/admin/keys/{key_id}", tags=["Admin Key Management"])
async def revoke_key_endpoint(key_id: int, admin_session: str = Depends(require_admin)):
    await revoke_api_key(key_id)
    return {"status": "ok", "message": f"API Key {key_id} đã bị thu hồi và xóa."}


# 5. Global API Key Authentication HTTP Middleware
PUBLIC_PATHS = {
    "/",
    "/health",
    "/api/v1/auth/login",
    "/api/v1/auth/logout",
    "/api/v1/auth/status",
}

PUBLIC_PREFIXES = [
    "/admin",
    "/admin/",
    "/events_images/",
    "/recordings/",
    "/docs",
    "/openapi.json",
    "/redoc",
]

def get_required_scope(method: str, path: str) -> str:
    if path in ("/detect", "/detect/batch", "/ws/stream", "/ws/rtsp") or path == "/streams" or path.startswith("/streams/"):
        return "inference"
    if method in ("POST", "PUT", "DELETE", "PATCH") and not path.startswith("/api/v1/tracking/search"):
        return "admin"
    return "data:read"

@app.middleware("http")
async def api_key_auth_middleware(request: Request, call_next):
    if os.getenv("REQUIRE_API_KEY", "false").lower() != "true" or request.method == "OPTIONS":
        return await call_next(request)
        
    path = request.url.path
    if path in PUBLIC_PATHS or any(path.startswith(p) for p in PUBLIC_PREFIXES):
        return await call_next(request)
        
    try:
        required_scope = get_required_scope(request.method, path)
        dependency = verify_api_key(required_scope)
        await dependency(request)
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Authentication error: {str(e)}"})
        
    return await call_next(request)

# Serve crop images so the browser gallery can display thumbnails
_EVENTS_DIR = "/events_images"
os.makedirs(_EVENTS_DIR, exist_ok=True)
app.mount("/events_images", StaticFiles(directory=_EVENTS_DIR), name="events_images")

_RECORDINGS_DIR = "/app/recordings"
os.makedirs(_RECORDINGS_DIR, exist_ok=True)


class HLSStaticFiles(StaticFiles):
    """StaticFiles subclass that injects correct MIME types for HLS playlists and segments.
    Without these overrides browsers receive application/octet-stream and refuse to play.
    """
    _HLS_TYPES = {
        ".m3u8": "application/vnd.apple.mpegurl",
        ".ts":   "video/mp2t",
        ".mp4":  "video/mp4",
    }

    async def get_response(self, path: str, scope):
        from starlette.responses import FileResponse
        resp = await super().get_response(path, scope)
        if isinstance(resp, FileResponse):
            for ext, mime in self._HLS_TYPES.items():
                if resp.path.endswith(ext):  # type: ignore[attr-defined]
                    resp.headers["Content-Type"] = mime
                    resp.headers["Cache-Control"] = "no-cache"
                    break
        return resp


app.mount("/recordings", HLSStaticFiles(directory=_RECORDINGS_DIR), name="recordings")


app.mount("/recordings", HLSStaticFiles(directory=_RECORDINGS_DIR), name="recordings")


# 6. Serve API Keys Management Dashboard at /admin
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

app.mount("/admin", StaticFiles(directory="/app/admin", html=True), name="admin")

@app.get("/", tags=["Web UI"])
async def root_redirect():
    return RedirectResponse(url="/admin/")


@app.get(
    "/api/recordings/list",
    tags=["Recording"],
    summary="List all HLS recording sessions",
)
async def recordings_list():
    """Returns all HLS recording sessions with segment info and live status."""
    try:
        sessions = []
        for entry in sorted(os.scandir(_RECORDINGS_DIR), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            m3u8 = os.path.join(entry.path, "live.m3u8")
            if not os.path.exists(m3u8):
                continue
            segments = sorted([f for f in os.listdir(entry.path) if f.endswith(".ts")])
            total_bytes = sum(
                os.path.getsize(os.path.join(entry.path, s)) for s in segments
            )
            # Consider recording "live" if the m3u8 was modified in the last 30 seconds
            m3u8_mtime = os.path.getmtime(m3u8)
            is_live = (time.time() - m3u8_mtime) < 30
            sessions.append({
                "stream_id": entry.name,
                "m3u8_url": f"/recordings/{entry.name}/live.m3u8",
                "segment_count": len(segments),
                "total_mb": round(total_bytes / 1024 / 1024, 1),
                "mtime": m3u8_mtime,
                "is_live": is_live,
            })
        return {"sessions": sessions, "count": len(sessions)}
    except Exception as e:
        raise HTTPException(500, f"Failed to list recordings: {e}")



# ═══════════════════════════════════════════════════════════════════
#  Health
# ═══════════════════════════════════════════════════════════════════

@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
    response_description="Server status, Triton readiness, encoder, concurrency limits.",
)
async def health():
    """Liveness + readiness. `triton_ready` must be true before inference."""
    triton_ready = await _effective_triton_ready()
    go2rtc_ready = await _go2rtc_ready()
    return {
        "status": "ok",
        "triton_ready": triton_ready,
        "encoder_loaded": encoder is not None,
        "max_concurrent": MAX_CONCURRENT,
        "max_fps": MAX_FPS,
        "max_batch_files": MAX_BATCH_FILES,
        "gpu_count": len(discover_gpus()),
        "nms_backend": nms_backend(),
        "rtsp_backend": _rtsp_effective_backend(),
        "rtsp_backend_requested": RTSP_BACKEND,
        "rtsp_gstreamer_available": _gstreamer_available(),
        "rtsp_gstreamer_decoder": _gstreamer_selected_decoder() if _gstreamer_available() else None,
        "rtsp_opencv_gstreamer_available": _opencv_has_gstreamer(),
        "rtsp_python_gstreamer_available": _python_gstreamer_available(),
        "go2rtc_enabled": _go2rtc_available(),
        "go2rtc_ready": go2rtc_ready,
        "go2rtc_public_url": GO2RTC_PUBLIC_URL or None,
        "require_api_key": os.getenv("REQUIRE_API_KEY", "false").lower() == "true",
    }


@app.get(
    "/go2rtc/status",
    tags=["Streams"],
    summary="go2rtc live-view status",
)
async def go2rtc_status():
    if not GO2RTC_API_URL:
        return {"enabled": False, "ready": False, "api_url": None, "streams": {}}
    try:
        streams = await _go2rtc_call("GET", "/api/streams")
        return {
            "enabled": True,
            "ready": True,
            "api_url": GO2RTC_API_URL,
            "public_url": GO2RTC_PUBLIC_URL or None,
            "streams": streams,
        }
    except Exception as exc:
        return {
            "enabled": True,
            "ready": False,
            "api_url": GO2RTC_API_URL,
            "public_url": GO2RTC_PUBLIC_URL or None,
            "error": str(exc),
        }



# ═══════════════════════════════════════════════════════════════════
#  go2rtc Reverse Proxy
#  The browser calls /go2rtc/... on the API server. The API forwards
#  the request to the internal go2rtc container (http://go2rtc:1984).
#  Works on localhost, LAN, Traefik HTTPS, Tailscale — zero extra config.
# ═══════════════════════════════════════════════════════════════════

@app.api_route(
    "/go2rtc/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    tags=["Streams"],
    summary="go2rtc API proxy",
    include_in_schema=False,
)
async def go2rtc_proxy(request: Request, path: str):
    """
    Transparent HTTP reverse proxy to go2rtc. The browser calls this endpoint
    instead of connecting to go2rtc port 1984 directly.
    Handles REST, SSE, and SDP negotiation (api/webrtc).
    WebSocket streams use the /go2rtc/ws/* endpoint.
    """
    from fastapi.responses import Response as FastAPIResponse, JSONResponse
    if not GO2RTC_API_URL:
        return JSONResponse({"error": "go2rtc is not configured on this server"}, status_code=503)

    target_url = f"{GO2RTC_API_URL}/{path}"
    params = str(request.url.query)
    if params:
        target_url = f"{target_url}?{params}"

    body = await request.body()
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=body,
        )

    excluded = {"transfer-encoding", "connection", "keep-alive"}
    proxy_headers = {k: v for k, v in resp.headers.items() if k.lower() not in excluded}
    return FastAPIResponse(
        content=resp.content,
        status_code=resp.status_code,
        headers=proxy_headers,
        media_type=resp.headers.get("content-type"),
    )


@app.websocket("/go2rtc/ws/{path:path}")
async def go2rtc_ws_proxy(websocket: WebSocket, path: str):
    if not await authenticate_websocket(websocket, required_scope="inference"):
        return
    """
    WebSocket reverse proxy to go2rtc.
    Used by the browser for go2rtc WebRTC signaling (/api/ws) and live streams.
    Browser connects to wss://triton-api.yourdomain.com/go2rtc/ws/api/ws
    which is forwarded to ws://go2rtc:1984/api/ws internally.
    """
    import websockets as _ws_lib

    if not GO2RTC_API_URL:
        await websocket.close(code=1011)
        return

    ws_base = GO2RTC_API_URL.replace("https://", "wss://").replace("http://", "ws://")
    params = str(websocket.url.query)
    target_ws_url = f"{ws_base}/{path}"
    if params:
        target_ws_url = f"{target_ws_url}?{params}"

    await websocket.accept()

    try:
        async with _ws_lib.connect(target_ws_url) as upstream:
            async def _browser_to_go2rtc():
                try:
                    while True:
                        try:
                            data = await websocket.receive_bytes()
                        except Exception:
                            try:
                                text = await websocket.receive_text()
                                await upstream.send(text)
                                continue
                            except Exception:
                                break
                        await upstream.send(data)
                except Exception:
                    pass

            async def _go2rtc_to_browser():
                try:
                    async for message in upstream:
                        if isinstance(message, bytes):
                            await websocket.send_bytes(message)
                        else:
                            await websocket.send_text(message)
                except Exception:
                    pass

            done, pending = await asyncio.wait(
                [
                    asyncio.ensure_future(_browser_to_go2rtc()),
                    asyncio.ensure_future(_go2rtc_to_browser()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
    except Exception as exc:
        logger.debug("go2rtc WS proxy closed: %s", exc)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.get(
    "/gpus",
    tags=["System"],
    summary="List GPUs and models per GPU",
    response_description="GPU list, models_by_gpu map, default_gpu index.",
)
async def gpus_list(refresh: bool = False):
    """
    List GPU indices visible to the API container.

    Use before upload (`gpus` form field) or `PUT /models/{name}/instances`.
    Set `refresh=true` to rescan hardware.
    """
    gpu_list = discover_gpus(refresh=refresh)
    by_gpu = models_per_gpu(MODEL_REPO_PATH)
    return {
        "gpus": gpu_list,
        "models_by_gpu": by_gpu,
        "default_gpu": gpu_list[0]["index"] if gpu_list else 0,
    }


@app.get(
    "/system/status",
    tags=["System"],
    summary="Host + Triton system status",
)
async def system_status():
    """
    Combined status: API health, Triton live/ready, host CPU/RAM,
    GPU utilization, Triton per-model stats, Prometheus metrics summary.
    """
    api_health = {
        "status": "ok",
        "triton_ready": await _effective_triton_ready(),
        "encoder_loaded": encoder is not None,
        "max_concurrent": MAX_CONCURRENT,
        "max_fps": MAX_FPS,
        "max_batch_files": MAX_BATCH_FILES,
        "gpu_count": len(discover_gpus()),
        "nms_backend": nms_backend(),
        "rtsp_backend": _rtsp_effective_backend(),
        "rtsp_backend_requested": RTSP_BACKEND,
        "rtsp_gstreamer_available": _gstreamer_available(),
        "rtsp_gstreamer_decoder": _gstreamer_selected_decoder() if _gstreamer_available() else None,
        "rtsp_opencv_gstreamer_available": _opencv_has_gstreamer(),
        "rtsp_python_gstreamer_available": _python_gstreamer_available(),
    }
    try:
        return await collect_system_status(api_health, MODEL_REPO_PATH)
    except Exception as exc:
        raise HTTPException(503, str(exc))


@app.get(
    "/system/triton/stats",
    tags=["System"],
    summary="Triton model inference statistics",
)
async def system_triton_stats():
    """Proxy Triton `GET /v2/models/stats` (inference counts, latency ns)."""
    import httpx

    url = f"{TRITON_HTTP_URL.rstrip('/')}/v2/models/stats"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=15.0)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        raise HTTPException(503, str(exc))


@app.get(
    "/system/metrics",
    tags=["System"],
    summary="Triton Prometheus metrics (raw)",
    response_description="text/plain Prometheus exposition from Triton :8002",
)
async def system_metrics():
    """
    Proxy Triton Prometheus metrics (default `TRITON_METRICS_URL`, port 8002).
    Same format as scraping Triton metrics directly.
    """
    try:
        text = await fetch_triton_metrics_raw()
    except Exception as exc:
        raise HTTPException(503, str(exc))
    if not text:
        raise HTTPException(503, "Triton metrics endpoint unavailable")
    from fastapi.responses import PlainTextResponse

    return PlainTextResponse(text, media_type="text/plain; version=0.0.4")


# ═══════════════════════════════════════════════════════════════════
#  Single-image inference
# ═══════════════════════════════════════════════════════════════════

# In-memory dictionary for stateless trackers (e.g. Webcam / Video upload via POST /detect)
# Format: { "stream_id": {"tracker": BYTETracker, "updated_at": float} }
stateless_trackers: dict[str, dict] = {}

@app.post(
    "/detect",
    tags=["Inference"],
    summary="Detect on one image",
    response_description="COCO instance JSON with annotations, inference_imgsz.",
)
async def detect(
    file: UploadFile = File(..., description="JPEG or PNG image"),
    model: str = Form(..., description="Triton model name"),
    prompts: Optional[str] = Form(
        None,
        description="YOLOE only: comma-separated text prompts (e.g. person,car). "
        "Does not affect standard YOLO labels.",
    ),
    classes: Optional[str] = Form(
        None,
        description="Standard YOLO only: optional override of labels.json. "
        "Ignored by YOLOE when prompts is set.",
    ),
    imgsz: Optional[str] = Form(
        None,
        description="Letterbox size: 640 (square), or height,width e.g. 1280,720. Default 640.",
    ),
    conf: float = Form(DEFAULT_CONF),
    iou: float = Form(DEFAULT_IOU),
    stream_id: Optional[str] = Form(None, description="Optional stream ID to maintain tracking state across requests."),
    track: bool = Form(False, description="Set to true to run tracking algorithm. Requires stream_id."),
):
    """
    Run inference on one image (multipart form).

  **Form fields:** `file`, `model`, optional `prompts` (YOLOE), `classes` (YOLO), `imgsz`, `conf`, `iou`.

  **YOLOE:** use `prompts=person,car` — not `classes` (avoids overwriting YOLO label maps in hybrid runs).

  **Response:** `image_id`, `image_shape`, `annotations[]` with `bbox` [x,y,w,h],
  `category_name`, `score`, optional `segmentation` (RLE), plus `inference_imgsz`.
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, "Empty image file")
    try:
        input_size = parse_imgsz(imgsz)
    except APIError as exc:
        if exc.status_code >= 500:
            try:
                await triton_unload_model(TRITON_HTTP_URL, model_name)
            except Exception:
                pass
            shutil.rmtree(os.path.join(MODEL_REPO_PATH, model_name), ignore_errors=True)
            _model_registry.pop(model_name, None)
            _encoders_by_model.pop(model_name, None)
            _bad_model_encoders.discard(model_name)
            _model_failures.pop(model_name, None)
        raise HTTPException(exc.status_code, exc.message)

    image_id = int(hashlib.md5(image_bytes[:128]).hexdigest()[:8], 16)

    try:
        result = await _run_inference(
            model, image_bytes, image_id, conf, iou, classes, prompts, input_size
        )
        result["inference_imgsz"] = list(input_size)
    except HTTPException:
        raise
    except APIError as exc:
        raise HTTPException(exc.status_code, exc.message)
    except Exception as exc:
        logger.exception(f"Inference failed for model '{model}'")
        raise HTTPException(500, str(exc))

    if track and stream_id:
        now = time.time()
        expired = [sid for sid, s in stateless_trackers.items() if now - s["updated_at"] > 60.0]
        for sid in expired:
            del stateless_trackers[sid]
            
        if stream_id not in stateless_trackers:
            stateless_trackers[stream_id] = {
                "tracker": BYTETracker(track_thresh=0.45, match_thresh=1.5, max_lost=30),
                "updated_at": now
            }
        
        tracker_state = stateless_trackers[stream_id]
        tracker_state["updated_at"] = now
        tracker = tracker_state["tracker"]
        
        byte_dets = []
        for ann in result.get("annotations", []):
            x, y, w, h = ann["bbox"]
            byte_dets.append({
                "bbox": [x, y, x + w, y + h],
                "score": float(ann.get("score", 1.0)),
            })
            
        if byte_dets or len(tracker.tracked) > 0 or len(tracker.lost) > 0:
            track_results = tracker.update(byte_dets)
            
            new_annotations = []
            for i, t in enumerate(track_results):
                ann = result["annotations"][i]
                if t is not None:
                    ann["local_id"] = t["local_id"]
                    ann["track_id"] = t["global_id"] or f"L{t['local_id']}"
                new_annotations.append(ann)
            result["annotations"] = new_annotations

    return JSONResponse(result)


@app.post(
    "/detect/batch",
    tags=["Inference"],
    summary="Detect on multiple images",
    response_description="results array with one COCO-style result per uploaded image.",
)
async def detect_batch(
    files: list[UploadFile] = File(..., description="JPEG/PNG images"),
    model: str = Form(..., description="Triton model name"),
    prompts: Optional[str] = Form(
        None,
        description="YOLOE only: comma-separated text prompts (e.g. person,car).",
    ),
    classes: Optional[str] = Form(
        None,
        description="Standard YOLO only: optional override of labels.json.",
    ),
    imgsz: Optional[str] = Form(
        None,
        description="Letterbox size: 640 or height,width. Default 640.",
    ),
    conf: float = Form(DEFAULT_CONF),
    iou: float = Form(DEFAULT_IOU),
):
    """
    Run inference on multiple uploaded images.

  This endpoint returns one result per image. Internally it reuses the same
  single-image inference path so output stays identical to `/detect`.
    """
    if not files:
        raise HTTPException(400, "No files uploaded")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(400, f"Too many files: max {MAX_BATCH_FILES}")

    try:
        input_size = parse_imgsz(imgsz)
    except APIError as exc:
        raise HTTPException(exc.status_code, exc.message)

    results: list[dict[str, Any]] = []
    for idx, file in enumerate(files):
        image_bytes = await file.read()
        if not image_bytes:
            results.append({
                "index": idx,
                "filename": file.filename,
                "error": "Empty image file",
            })
            continue
        image_id = int(hashlib.md5(image_bytes[:128]).hexdigest()[:8], 16)
        try:
            result = await _run_inference(
                model, image_bytes, image_id, conf, iou, classes, prompts, input_size
            )
            result["index"] = idx
            result["filename"] = file.filename
            result["inference_imgsz"] = list(input_size)
            results.append(result)
        except HTTPException as exc:
            results.append({
                "index": idx,
                "filename": file.filename,
                "error": exc.detail,
                "status_code": exc.status_code,
            })
        except APIError as exc:
            results.append({
                "index": idx,
                "filename": file.filename,
                "error": exc.message,
                "status_code": exc.status_code,
            })
        except Exception as exc:
            logger.exception(f"Batch inference failed for model '{model}' file '{file.filename}'")
            results.append({
                "index": idx,
                "filename": file.filename,
                "error": str(exc),
                "status_code": 500,
            })

    ok = sum(1 for r in results if "error" not in r)
    return JSONResponse({
        "model": model,
        "count": len(results),
        "ok": ok,
        "failed": len(results) - ok,
        "results": results,
    })


# ═══════════════════════════════════════════════════════════════════
#  Live-stream WebSocket
# ═══════════════════════════════════════════════════════════════════

@app.post(
    "/rtsp/probe",
    tags=["Inference"],
    summary="Probe an RTSP camera",
    response_description="Connection status and first-frame metadata.",
)
async def rtsp_probe(req: RTSPProbeRequest):
    """
    Open an RTSP/RSTPS URL and read one frame.

    Use this before `WS /ws/rtsp` to check credentials, network reachability,
    and camera resolution. The API never stores the URL.
    """
    try:
        url = _validate_rtsp_url(req.url)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    loop = asyncio.get_event_loop()
    cap = None
    try:
        cap = await asyncio.wait_for(
            loop.run_in_executor(None, _open_rtsp_capture, url),
            timeout=RTSP_OPEN_TIMEOUT_MS / 1000.0 + 2.0
        )
        _, (h, w) = await loop.run_in_executor(
            None, _read_rtsp_jpeg, cap, _clamp_jpeg_quality(req.jpeg_quality)
        )
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        return {
            "ok": True,
            "url_scheme": urlparse(url).scheme,
            "width": int(w),
            "height": int(h),
            "source_fps": round(float(src_fps), 3) if src_fps > 0 else None,
            "max_output_fps": MAX_FPS,
            "jpeg_quality": _clamp_jpeg_quality(req.jpeg_quality),
            "rtsp_backend": _rtsp_effective_backend(),
            "rtsp_backend_requested": RTSP_BACKEND,
            "rtsp_gstreamer_available": _gstreamer_available(),
            "rtsp_opencv_gstreamer_available": _opencv_has_gstreamer(),
            "rtsp_python_gstreamer_available": _python_gstreamer_available(),
            "nvidia_gpu_detected": _has_nvidia_gpu(),
        }
    except Exception as exc:
        raise HTTPException(503, str(exc))
    finally:
        if cap is not None:
            await loop.run_in_executor(None, cap.release)


@app.websocket(
    "/ws/rtsp",
    name="rtsp_stream",
)
async def ws_rtsp(
    websocket: WebSocket,
    url: str,
    fps: float = MAX_FPS,
    jpeg_quality: int = RTSP_JPEG_QUALITY,
):
    if not await authenticate_websocket(websocket, required_scope="inference"):
        return
    """
    RTSP bridge over WebSocket.

    **Query params:** `url` (`rtsp://` or `rtsps://`), `fps`, `jpeg_quality`.

    **Protocol:** server reads the RTSP camera and sends raw **JPEG bytes** to
    the client. This endpoint does not run inference by itself; browser clients
    can forward these JPEG bytes to `/ws/stream`.
    """
    await websocket.accept()
    try:
        rtsp_url = _validate_rtsp_url(url)
    except ValueError as exc:
        await websocket.close(code=1008, reason=str(exc))
        return

    out_fps = _clamp_rtsp_fps(fps)
    quality = _clamp_jpeg_quality(jpeg_quality)
    interval = 1.0 / out_fps
    loop = asyncio.get_event_loop()
    cap = None
    frame_idx = 0
    failures = 0
    logger.info(
        f"RTSP bridge started — fps={out_fps} quality={quality} host={urlparse(rtsp_url).hostname}"
    )

    try:
        cap = await asyncio.wait_for(
            loop.run_in_executor(None, _open_rtsp_capture, rtsp_url),
            timeout=RTSP_OPEN_TIMEOUT_MS / 1000.0 + 2.0
        )
        next_send = time.perf_counter()
        while True:
            now = time.perf_counter()
            if now < next_send:
                await asyncio.sleep(next_send - now)

            try:
                jpeg, _ = await loop.run_in_executor(None, _read_rtsp_jpeg, cap, quality)
                failures = 0
            except Exception as exc:
                failures += 1
                if failures >= RTSP_MAX_READ_FAILURES:
                    logger.warning(f"RTSP bridge stopping after read failures: {exc}")
                    await websocket.close(code=1011, reason=str(exc))
                    return
                await asyncio.sleep(min(interval, 0.2))
                continue

            await websocket.send_bytes(jpeg)
            frame_idx += 1
            next_send = time.perf_counter() + interval

    except WebSocketDisconnect:
        logger.info(f"RTSP bridge disconnected after {frame_idx} frames")
    except Exception as exc:
        logger.warning(f"RTSP bridge error: {exc}")
        try:
            await websocket.close(code=1011, reason=str(exc))
        except Exception:
            pass
    finally:
        if cap is not None:
            await loop.run_in_executor(None, cap.release)


@app.post(
    "/streams",
    tags=["Streams"],
    summary="Create a production RTSP stream worker",
)
async def streams_create(req: ManagedStreamCreate, request: Request):
    """
    API-owned RTSP stream. The API connects to the RTSP camera, runs inference
    server-side, and exposes:

    - `WS /streams/{id}/events` — detection JSON (always)
    - `WS /streams/{id}/preview` — JPEG preview (`annotated_preview=true` draws boxes on the exact inference frame;
      `false` sends raw frames for client-side overlay)

    If a stream with the same RTSP URL is already active, the existing stream
    is returned instead of opening a duplicate connection to the camera.
    """
    if not req.models:
        if req.enable_tracking:
            req.models = ["person"]
        else:
            raise HTTPException(400, "models must contain at least one model")
    for model in req.models:
        validate_model_name(model)
        if not _model_exists_on_disk(model):
            raise HTTPException(404, f"Model not found: {model}")
    try:
        _validate_rtsp_url(req.url)
        parse_imgsz(req.imgsz)
    except (ValueError, APIError) as exc:
        raise HTTPException(400, str(getattr(exc, "message", exc)))
    live_transport = (req.live_transport or "go2rtc").strip().lower()
    if live_transport not in {"go2rtc", "api_jpeg"}:
        raise HTTPException(400, "live_transport must be go2rtc or api_jpeg")

    # ── Dedup: reuse existing stream ONLY if exact matching RTSP URL, models, classes, prompts, tracking and session ──
    canonical_url = req.url.rstrip("/")
    client_ip = await get_request_session_identifier(request)
    req_models = sorted(list(dict.fromkeys(req.models or [])))
    req_classes = req.classes
    req_prompts = req.prompts
    req_tracking = bool(getattr(req, "enable_tracking", False))

    for existing in managed_streams.values():
        existing_url = getattr(existing, "_original_url", existing.url).rstrip("/")
        existing_models = sorted(list(dict.fromkeys(existing.requested_models or [])))
        existing_classes = existing.classes
        existing_prompts = existing.prompts
        existing_tracking = bool(getattr(existing, "tracking_enabled", False))
        existing_ip = getattr(existing, "client_ip", "unknown")

        if (
            existing_url == canonical_url
            and existing_models == req_models
            and existing_classes == req_classes
            and existing_prompts == req_prompts
            and existing_tracking == req_tracking
            and existing_ip == client_ip
        ):
            logger.info(
                "streams_create: reusing exact matching stream %s for URL %s (models=%s, session=%s)",
                existing.id, canonical_url, req_models, client_ip
            )
            patch_data = {k: v for k, v in req.dict(exclude_unset=True).items() if v is not None}
            existing.patch(ManagedStreamPatch(**patch_data))
            return existing.snapshot()

    # ── Derive a stable, configuration-aware stream_id from URL + models + classes + prompts + tracking + session ──
    import hashlib
    key_str = f"{canonical_url}|{','.join(req_models)}|{req_classes or ''}|{req_prompts or ''}|{req_tracking}|{client_ip}"
    stream_id = hashlib.sha1(key_str.encode()).hexdigest()[:12]

    # If a stale entry with same id exists (stopped), remove it first
    if stream_id in managed_streams:
        old = managed_streams.pop(stream_id)
        await old.stop()

    stream = ManagedRTSPStream(stream_id, req)
    # Store original URL before go2rtc rewrites it
    stream._original_url = canonical_url
    # Record the creating client's API key/IP for per-session gallery separation
    stream.client_ip = client_ip
    stream.live_transport = live_transport
    if _go2rtc_available():
        stream.go2rtc_name = _go2rtc_stream_name(stream_id)
        stream.go2rtc_public_url = _go2rtc_public_base(request)
        try:
            # Always register using the original camera URL, never the loopback URL
            orig_url = getattr(stream, "_original_url", stream.url)
            await _go2rtc_register_stream(stream.go2rtc_name, orig_url)
            # Use go2rtc as the RTSP fan-out source for API inference too.
            # This avoids opening multiple sessions to the physical camera
            # when the browser also consumes WebRTC from go2rtc.
            stream.url = f"rtsp://go2rtc:8554/{stream.go2rtc_name}_raw"
            stream.go2rtc_error = None
        except Exception as exc:
            stream.go2rtc_error = str(exc)
            if live_transport == "go2rtc":
                stream.live_transport = "api_jpeg"
            logger.warning("go2rtc registration failed for stream %s: %s", stream_id, exc)
    managed_streams[stream_id] = stream
    stream.start()

    # Save/upsert to persistent_streams SQLite table for NVR stream persistence
    try:
        from auth import db_execute, db_fetch_one

        # Resolve api_key_id: look up the numeric PK from the API key header (if authenticated via key)
        api_key_id_val = None
        _req_api_key = (
            request.headers.get("X-API-Key")
            or (request.headers.get("Authorization", "")[7:].strip()
                if request.headers.get("Authorization", "").lower().startswith("bearer ") else None)
            or request.query_params.get("api_key")
        )
        if _req_api_key:
            import hashlib as _hl
            _hashed = _hl.sha256(_req_api_key.encode()).hexdigest()
            _key_row = await db_fetch_one("SELECT id FROM api_keys WHERE key_hash = ?", (_hashed,))
            if _key_row:
                api_key_id_val = _key_row["id"]

        await db_execute(
            """INSERT INTO persistent_streams 
               (stream_id, name, url, models, classes, prompts, source_type, imgsz, conf, fps, preview_fps, source_max_height, enable_tracking, enable_recording, rec_format, overlay_mode, client_ip, live_transport, api_key_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(stream_id) DO UPDATE SET
                 name=excluded.name, url=excluded.url, models=excluded.models, classes=excluded.classes,
                 prompts=excluded.prompts, source_type=excluded.source_type, imgsz=excluded.imgsz,
                 conf=excluded.conf, fps=excluded.fps, preview_fps=excluded.preview_fps,
                 source_max_height=excluded.source_max_height, enable_tracking=excluded.enable_tracking,
                 enable_recording=excluded.enable_recording, rec_format=excluded.rec_format,
                 overlay_mode=excluded.overlay_mode, client_ip=excluded.client_ip,
                 live_transport=excluded.live_transport, api_key_id=excluded.api_key_id""",
            (
                stream_id, req.name or stream_id, canonical_url,
                ",".join(req_models), req_classes, req_prompts,
                getattr(req, "source_type", None) or ("rtsp" if canonical_url.startswith(("rtsp://", "rtsps://", "http://", "https://")) else "webcam"),
                req.imgsz or 640, req.conf if req.conf is not None else 0.5,
                req.fps or 30, req.preview_fps or 10, req.source_max_height or 720,
                1 if req_tracking else 0, 1 if getattr(req, "enable_recording", False) else 0,
                getattr(req, "rec_format", "hls") or "hls", getattr(req, "overlay_mode", "exact") or "exact",
                client_ip, live_transport, api_key_id_val
            )
        )
    except Exception as exc:
        logger.warning("Failed to save stream %s to DB: %s", stream_id, exc)

    return stream.snapshot()



@app.get(
    "/streams",
    tags=["Streams"],
    summary="List managed RTSP streams",
)
async def streams_list():
    return {"streams": [stream.snapshot() for stream in managed_streams.values()]}


@app.get(
    "/streams/{stream_id}",
    tags=["Streams"],
    summary="Get managed RTSP stream status",
)
async def streams_get(stream_id: str):
    stream = managed_streams.get(stream_id)
    if stream:
        return stream.snapshot()
    raise HTTPException(404, "Stream not found")


@app.patch(
    "/streams/{stream_id}",
    tags=["Streams"],
    summary="Update managed RTSP stream parameters",
)
async def streams_patch(stream_id: str, req: ManagedStreamPatch):
    stream = managed_streams.get(stream_id)
    if not stream:
        raise HTTPException(404, "Stream not found")
    if req.models is not None:
        for model in req.models:
            validate_model_name(model)
            if not _model_exists_on_disk(model):
                raise HTTPException(404, f"Model not found: {model}")
    try:
        return stream.patch(req)
    except APIError as exc:
        raise HTTPException(400, exc.message)
class RecordingToggleRequest(BaseModel):
    enabled: bool


@app.post(
    "/api/v1/streams/{stream_id}/recording",
    tags=["Streams"],
    summary="Toggle server-side raw video recording",
)
async def toggle_stream_recording(stream_id: str, req: RecordingToggleRequest):
    stream = managed_streams.get(stream_id)
    if not stream:
        raise HTTPException(404, "Stream not found")
    stream.recording_enabled = req.enabled
    return {"status": "success", "recording_enabled": stream.recording_enabled}


@app.get(
    "/api/v1/recordings",
    tags=["Streams"],
    summary="List all server-side recordings",
)
@app.get(
    "/api/v1/recordings/{stream_id}",
    tags=["Streams"],
    summary="List server-side recordings for a stream or all streams",
)
async def list_stream_recordings(stream_id: str = "all"):
    import os, glob
    from datetime import datetime
    rec_dirs = ["/app/recordings", "recordings"]
    rec_dir = next((d for d in rec_dirs if os.path.exists(d)), "/app/recordings")
    if not os.path.exists(rec_dir):
        return {"recordings": []}
        
    recordings = []
    is_all = not stream_id or stream_id.lower() in ("all", "all_streams", "*")
    
    # 1. Look for HLS directories
    pattern = "*" if is_all else f"{stream_id}_*"
    hls_dirs = glob.glob(os.path.join(rec_dir, pattern))
    for d in sorted(hls_dirs, reverse=True):
        if not os.path.isdir(d):
            continue
        m3u8_file = os.path.join(d, "live.m3u8")
        if not os.path.exists(m3u8_file):
            continue
            
        dir_name = os.path.basename(d)
        
        # Calculate total size of the directory
        total_size = 0
        try:
            for entry in os.scandir(d):
                if entry.is_file():
                    total_size += entry.stat().st_size
        except Exception:
            pass
                
        try:
            parts = dir_name.split("_")
            ts = int(parts[-1])
            dt = datetime.fromtimestamp(ts).isoformat()
        except Exception:
            try:
                mtime = os.path.getmtime(d)
                dt = datetime.fromtimestamp(mtime).isoformat()
            except Exception:
                dt = None
            
        recordings.append({
            "filename": dir_name,
            "is_hls": True,
            "size_bytes": total_size,
            "created_at": dt,
            "url": f"/recordings/{dir_name}/live.m3u8"
        })
        
    # 2. Look for video files (.mp4, .mkv, .ts)
    file_pattern = "*.*" if is_all else f"{stream_id}_*.*"
    files = glob.glob(os.path.join(rec_dir, file_pattern))
    for f in sorted(files, reverse=True):
        if not f.lower().endswith((".mp4", ".mkv", ".ts")):
            continue
        filename = os.path.basename(f)
        stats = os.stat(f)
        try:
            parts = filename.replace(".mp4", "").split("_")
            ts = int(parts[-1])
            dt = datetime.fromtimestamp(ts).isoformat()
        except Exception:
            dt = datetime.fromtimestamp(stats.st_mtime).isoformat()
        recordings.append({
            "filename": filename,
            "is_hls": False,
            "size_bytes": stats.st_size,
            "created_at": dt,
            "url": f"/recordings/{filename}"
        })
        
    recordings.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return {"recordings": recordings}


@app.delete(
    "/api/v1/recordings/{filename}",
    tags=["Streams"],
    summary="Delete a server-side recording file or directory",
)
async def delete_recording(filename: str):
    import os, shutil
    safe_name = os.path.basename(filename)
    rec_dirs = ["/app/recordings", "recordings"]
    deleted = False
    for rec_dir in rec_dirs:
        filepath = os.path.join(rec_dir, safe_name)
        if os.path.exists(filepath):
            try:
                if os.path.isdir(filepath):
                    shutil.rmtree(filepath)
                else:
                    os.remove(filepath)
                deleted = True
            except Exception as e:
                raise HTTPException(500, f"Failed to delete recording: {e}")
    if deleted:
        return {"status": "deleted", "filename": safe_name}
    raise HTTPException(404, "Recording not found")


@app.delete(
    "/streams/{stream_id}",
    tags=["Streams"],
    summary="Stop and remove a managed RTSP stream",
)
async def streams_delete(stream_id: str):
    stream = managed_streams.get(stream_id)
    if not stream:
        raise HTTPException(404, "Stream not found")
        
    async def _delayed_delete():
        await asyncio.sleep(1.0) # give websockets time to disconnect
        managed_streams.pop(stream_id, None)
        await stream.stop()
        await _go2rtc_unregister_stream(stream.go2rtc_name)
        try:
            from auth import db_execute
            await db_execute("DELETE FROM persistent_streams WHERE stream_id = ?", (stream_id,))
        except Exception as exc:
            logger.warning("Failed to delete stream %s from DB: %s", stream_id, exc)
            
    asyncio.create_task(_delayed_delete())
    return {"deleted": stream_id, "status": "scheduled"}


@app.websocket(
    "/streams/{stream_id}/events",
    name="stream_events",
)
async def streams_events(websocket: WebSocket, stream_id: str):
    if not await authenticate_websocket(websocket, required_scope="data:read"):
        return
    """Subscribe to server-side inference JSON for a managed RTSP stream."""
    stream = managed_streams.get(stream_id)
    if not stream:
        await websocket.close(code=1008, reason="Stream not found")
        return
    await websocket.accept()
    q = stream.subscribe_events()
    try:
        await websocket.send_json({"type": "status", "stream": stream.snapshot()})
        while True:
            event = await q.get()
            if isinstance(event, dict) and event.get("type") == "closed":
                await websocket.send_json(event)
                await websocket.close(code=1000, reason="Stream stopped")
                break
            await websocket.send_json({"type": "detections", **event})
    except WebSocketDisconnect:
        pass
    finally:
        stream.unsubscribe_events(q)


@app.websocket(
    "/streams/{stream_id}/preview",
    name="stream_preview",
)
async def streams_preview(websocket: WebSocket, stream_id: str, metadata: bool = False):
    if not await authenticate_websocket(websocket, required_scope="inference"):
        return
    """Subscribe to optional JPEG preview for a managed RTSP stream."""
    stream = managed_streams.get(stream_id)
    if not stream:
        await websocket.close(code=1008, reason="Stream not found")
        return
    await websocket.accept()
    q = stream.subscribe_preview()
    try:
        while True:
            item = await q.get()
            if item is None:
                await websocket.close(code=1000, reason="Stream stopped")
                break
            if isinstance(item, dict):
                jpeg = item.get("jpeg") or b""
                if metadata:
                    await websocket.send_json({
                        "type": "preview",
                        "frame_seq": item.get("frame_seq"),
                        "image_shape": item.get("image_shape"),
                        "ts": item.get("ts"),
                        "image_b64": base64.b64encode(jpeg).decode("ascii"),
                    })
                else:
                    await websocket.send_bytes(jpeg)
            else:
                await websocket.send_bytes(item)
    except WebSocketDisconnect:
        pass
    finally:
        stream.unsubscribe_preview(q)


@app.websocket(
    "/ws/stream",
    name="stream",
)
async def ws_stream(
    websocket: WebSocket,
    model: str,
    prompts: Optional[str] = None,
    classes: Optional[str] = None,
    imgsz: Optional[str] = None,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    fps: float = MAX_FPS,
    track: bool = False,
):
    if not await authenticate_websocket(websocket, required_scope="inference"):
        return
    """
    Live inference over WebSocket.

  **Query params:** `model`, `prompts` (YOLOE), `classes` (YOLO), `imgsz`, `conf`, `iou`, `fps`.

  **Protocol:** client sends raw **JPEG bytes** per frame; server replies with COCO JSON.
  Frames above `fps` return `{"dropped": true}` so clients can keep sending.
  YOLOE / hybrid-with-YOLOE requires `prompts` query param.
    """
    await websocket.accept()
    stream_id = str(id(websocket))
    limiter = StreamRateLimiter(fps)
    info = _model_info(model)

    needs_prompts = info["type"] == MODEL_TYPE_YOLOE_DYNAMIC
    if info["type"] == MODEL_TYPE_ENSEMBLE and info.get("ensemble_kind") == "hybrid":
        from model_detector import MODEL_TYPE_YOLOE_DYNAMIC as _YOLOE

        needs_prompts = any(
            s.get("model_type") == _YOLOE for s in info.get("steps", [])
        )
    if needs_prompts:
        if not parse_text_prompts(prompts, classes):
            await websocket.close(
                code=1008,
                reason="YOLOE / hybrid ensemble requires 'prompts' query param",
            )
            return
        try:
            if info["type"] == MODEL_TYPE_YOLOE_DYNAMIC:
                _encoder_for_model(model)
        except HTTPException as exc:
            await websocket.close(code=1011, reason=exc.detail)
            return

    try:
        input_size = parse_imgsz(imgsz)
    except APIError as exc:
        await websocket.close(code=1008, reason=exc.message)
        return

    frame_idx = 0
    logger.info(
        f"WebSocket stream {stream_id} started — model={model} fps={fps} imgsz={input_size}"
    )

    # Extract API key/IP session identifier from WebSocket handshake headers
    client_ip = await get_request_session_identifier(websocket)

    tracker = BYTETracker(track_thresh=0.45, match_thresh=1.5, max_lost=30) if track else None
    # Tracks already registered in DB (local_id set) — avoid duplicate saves
    _registered: set[int] = set()
    _ws_bbox_trail: dict[int, dict] = {} # {lid: {"pts": [...], "last_ts": float, "point_id": str|None, "cls": str}}

    try:
        while True:
            image_bytes: bytes = await websocket.receive_bytes()

            if not limiter.should_process(stream_id):
                await websocket.send_json({"dropped": True, "frame": frame_idx})
                continue

            try:
                result = await _run_inference(
                    model, image_bytes, frame_idx, conf, iou, classes, prompts, input_size
                )

                if tracker is not None:
                    byte_dets = []
                    for ann in result.get("annotations", []):
                        x, y, w, h = ann["bbox"]
                        byte_dets.append({
                            "bbox": [x, y, x + w, y + h],
                            "score": float(ann.get("score", 1.0)),
                        })
                    track_results = tracker.update(byte_dets)

                    for i, t in enumerate(track_results):
                        if i >= len(result["annotations"]) or t is None:
                            continue
                        lid = t["local_id"]
                        ann_meta = result["annotations"][i]
                        class_name = ann_meta.get("category_name") or ann_meta.get("label") or "object"
                        trail_bbox = [float(v) for v in t["bbox"]]
                        now_ts = time.time()

                        track_obj = tracker.get_track(lid)
                        if track_obj:
                            score_val = float(ann_meta.get("score", 1.0))
                            track_obj.add_crop_candidate(image_bytes, t["bbox"], score_val, now_ts)

                        # Accumulate path trail nodes (when center moves >= 3.0px)
                        ws_score = float(ann_meta.get("score", 1.0))
                        if ws_score >= 0.20:
                            ws_tr_id = t.get("track_id") or lid
                            is_new_ws_track = (
                                lid not in _ws_bbox_trail
                                or _ws_bbox_trail[lid].get("track_id") != ws_tr_id
                            )
                            if is_new_ws_track:
                                _ws_bbox_trail[lid] = {
                                    "track_id": ws_tr_id,
                                    "pts": [trail_bbox],
                                    "cls": class_name,
                                    "last_ts": now_ts,
                                    "point_id": None,
                                }
                            entry = _ws_bbox_trail[lid]
                            pts = entry["pts"]
                            if not is_new_ws_track and pts:
                                prev = pts[-1]
                                # Ground position (bottom-center of bounding box)
                                prev_cx, prev_feet_y = (prev[0] + prev[2]) / 2, prev[3]
                                cur_cx, cur_feet_y   = (trail_bbox[0] + trail_bbox[2]) / 2, trail_bbox[3]
                                moved = ((cur_cx - prev_cx)**2 + (cur_feet_y - prev_feet_y)**2) ** 0.5
                                dt = now_ts - entry.get("last_ts", 0.0)
                                if moved > 120.0:
                                    # Outlier / Teleportation filter: ignore sudden large jumps across frame
                                    pass
                                elif moved >= 3.0 or dt >= 0.2:
                                    pts.append(trail_bbox)
                                    entry["last_ts"] = now_ts
                                    if len(pts) > 100:
                                        entry["pts"] = [pts[0]] + pts[-99:]
                                    if entry.get("point_id"):
                                        asyncio.create_task(update_object_event_trail(entry["point_id"], entry["pts"]))

                        result["annotations"][i]["local_id"] = lid
                        result["annotations"][i]["track_id"] = t["global_id"] or f"L{lid}"

                        # Fast Re-ID & Gallery trigger at hit >= 3
                        hits = t["hits"]
                        if hits >= 3 and lid not in _registered and not t["global_id"]:
                            _registered.add(lid)
                            logger.info(f"[ReID WS] Fast trigger for lid={lid} hits={hits} class={class_name}")
                            try:
                                frame_bgr = _decode_jpeg_frame(image_bytes)
                                emb = await extract_embedding(frame_bgr, t["bbox"])
                                
                                gid = None
                                match_pid = None
                                is_new = False
                                reid_thresh = float(os.getenv("REID_THRESHOLD", "0.85"))
                                max_reid_age_s = float(os.getenv("MAX_REID_AGE_SECONDS", "4.0"))

                                if emb and class_name == "person":
                                    match = await search_object_with_meta(emb, class_name=class_name, threshold=reid_thresh)
                                    if match:
                                        _mts = match.get("timestamp")
                                        _reject_ws = False
                                        if _mts:
                                            try:
                                                from datetime import datetime, timezone
                                                _mt = datetime.fromisoformat(_mts.replace("Z", "+00:00"))
                                                _nt = datetime.now(timezone.utc)
                                                if (_nt - _mt).total_seconds() > max_reid_age_s:
                                                    _reject_ws = True
                                            except Exception:
                                                pass
                                        if not _reject_ws:
                                            gid = match.get("global_id")
                                            match_pid = match.get("id")
                                            logger.info(f"[ReID WS MATCH] Matched existing person {gid} (similarity >= {reid_thresh})")

                                prefix = class_name[:3].upper() if class_name else "OBJ"
                                if not gid:
                                    gid = f"{prefix}-{uuid.uuid4().hex[:6].upper()}"
                                    is_new = True

                                x1, y1, x2, y2 = map(int, t["bbox"])
                                h_f, w_f = (frame_bgr.shape[0], frame_bgr.shape[1]) if frame_bgr is not None else (1080, 1920)
                                norm_bbox = [round(x1/w_f, 4), round(y1/h_f, 4), round(x2/w_f, 4), round(y2/h_f, 4)]
                                norm_trail = [
                                    [round(pt[0]/w_f, 4), round(pt[1]/h_f, 4), round(pt[2]/w_f, 4), round(pt[3]/h_f, 4)]
                                    for pt in entry["pts"] if len(pt) >= 4
                                ]
                                if is_new:
                                    img_path = f"/events_images/{gid}_{int(time.time())}.jpg"
                                    img_path_full = f"/events_images/{gid}_{int(time.time())}_full.jpg"
                                    crop = frame_bgr[max(0,y1):max(0,y2), max(0,x1):max(0,x2)] if frame_bgr is not None else None
                                    if crop is not None and crop.size > 0:
                                        cv2.imwrite(img_path, crop)
                                    if frame_bgr is not None and frame_bgr.size > 0:
                                        cv2.imwrite(img_path_full, frame_bgr)
                                    pid = await add_object_event(
                                        global_id=gid,
                                        embedding=emb or [0.0] * 512,
                                        class_name=class_name,
                                        camera_id=f"ws_{stream_id}",
                                        image_path=img_path,
                                        client_ip=client_ip,
                                        image_path_full=img_path_full,
                                        bbox=[float(x1), float(y1), float(x2), float(y2)],
                                        track_session_id=str(lid),
                                        bbox_trail=entry["pts"],
                                    )
                                    entry["point_id"] = pid
                                else:
                                    entry["point_id"] = None
                                    if match_pid:
                                        now_iso = __import__('datetime').datetime.utcnow().isoformat() + 'Z'
                                        asyncio.create_task(update_object_last_seen(match_pid, now_iso))
                                        # Thumbnail & Trail intentionally NOT updated — original detection frame & trail remain pure

                                tracker.set_global_id(lid, gid)
                                result["annotations"][i]["track_id"] = gid
                            except Exception as reid_err:
                                logger.warning(f"[ReID WS] ERROR lid={lid}: {reid_err}", exc_info=True)

                    # Flush removed tracks and update best crop image
                    if hasattr(tracker, "removed") and tracker.removed:
                        for rm_t in list(tracker.removed):
                            rm_lid = rm_t.local_id
                            rm_entry = _ws_bbox_trail.pop(rm_lid, None)
                            if rm_entry and rm_entry.get("point_id"):
                                pid = rm_entry["point_id"]
                                asyncio.create_task(update_object_event_trail(pid, rm_entry["pts"]))
                                if rm_t.crops:
                                    try:
                                        best_c = rm_t.crops[0]
                                        best_q = float(best_c.get("quality", 0.0))
                                        if best_q > rm_entry.get("best_quality", 0.0):
                                            best_f = _decode_jpeg_frame(best_c["frame_jpeg"])
                                            gid = rm_t.global_id
                                            if best_f is not None and best_f.size > 0 and gid:
                                                x1, y1, x2, y2 = map(int, best_c["bbox"])
                                                crop_i = best_f[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                                                if crop_i.size > 0:
                                                    ts_slug = int(time.time() * 1000)
                                                    img_p = f"/events_images/{gid}_{ts_slug}_best.jpg"
                                                    img_pf = f"/events_images/{gid}_{ts_slug}_best_full.jpg"
                                                    cv2.imwrite(img_p, crop_i)
                                                    cv2.imwrite(img_pf, best_f)
                                                    asyncio.create_task(update_object_event_image(
                                                        pid, img_p, img_pf,
                                                        bbox=[float(x1), float(y1), float(x2), float(y2)],
                                                        quality=best_q
                                                    ))
                                    except Exception as img_err:
                                        logger.warning(f"Failed best crop update WS L{rm_lid}: {img_err}")
                        tracker.removed = []

                result["inference_imgsz"] = list(input_size)
                await websocket.send_json(result)
                frame_idx += 1

            except WebSocketDisconnect:
                raise
            except Exception as exc:
                logger.error(f"Stream {stream_id} frame {frame_idx} error: {exc}")
                try:
                    await websocket.send_json({"error": str(exc), "frame": frame_idx})
                except WebSocketDisconnect:
                    raise
                except RuntimeError:
                    break
    except WebSocketDisconnect:
        logger.info(f"Stream {stream_id} disconnected after {frame_idx} frames")
    finally:
        # Stream finished/disconnected — flush final trails & best crops to Qdrant
        for lid, entry in _ws_bbox_trail.items():
            if entry.get("point_id"):
                pid = entry["point_id"]
                if entry.get("pts"):
                    try:
                        await update_object_event_trail(pid, entry["pts"])
                    except Exception:
                        pass
                if tracker:
                    trk_obj = tracker.get_track(lid)
                    if trk_obj and trk_obj.crops:
                        try:
                            best_c = trk_obj.crops[0]
                            best_f = _decode_jpeg_frame(best_c["frame_jpeg"])
                            gid = trk_obj.global_id
                            if best_f is not None and best_f.size > 0 and gid:
                                x1, y1, x2, y2 = map(int, best_c["bbox"])
                                crop_i = best_f[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                                img_p = f"/events_images/{gid}_final.jpg"
                                img_pf = f"/events_images/{gid}_final_full.jpg"
                                if crop_i.size > 0:
                                    cv2.imwrite(img_p, crop_i)
                                    cv2.imwrite(img_pf, best_f)
                                    # Thumbnail intentionally NOT updated — original first-detection crop is permanent
                        except Exception as img_err:
                            logger.warning(f"Failed final crop update on disconnect WS L{lid}: {img_err}")
        limiter.remove_stream(stream_id)


# ═══════════════════════════════════════════════════════════════════
#  Tracking / Object Re-ID
# ═══════════════════════════════════════════════════════════════════

async def get_request_session_identifier(request_or_websocket) -> str:
    """Returns the API key identifier (prefix + name) or falls back to client IP."""
    headers = request_or_websocket.headers
    api_key = headers.get("X-API-Key")
    if not api_key:
        auth_header = headers.get("Authorization")
        if auth_header and auth_header.lower().startswith("bearer "):
            api_key = auth_header[7:].strip()
    if not api_key:
        query_params = request_or_websocket.query_params
        api_key = query_params.get("api_key") or query_params.get("token")
        
    if api_key:
        import hashlib
        hashed = hashlib.sha256(api_key.encode()).hexdigest()
        from auth import db_fetch_one
        try:
            row = await db_fetch_one("SELECT prefix, key_name FROM api_keys WHERE key_hash = ?", (hashed,))
            if row:
                return f"{row['prefix']} ({row['key_name']})"
        except Exception:
            pass
        return api_key[:12] if len(api_key) >= 12 else api_key
        
    forwarded_for = headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    client = getattr(request_or_websocket, "client", None)
    if client and getattr(client, "host", None):
        return client.host
    return "unknown"


def _get_client_ip(request: Request) -> str:
    """Extract client IP, respecting reverse-proxy X-Forwarded-For header."""
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host or "unknown"


async def _determine_session_filter(request: Request, session: Optional[str] = None) -> tuple[Optional[str], bool]:
    """Helper to determine the session/client_ip filter based on authentication.
    Returns (resolved_session_filter_string, is_admin_boolean).
    """
    from auth import validate_admin_session
    is_admin = False
    session_id = request.cookies.get("session_id")
    if session_id and await validate_admin_session(session_id):
        is_admin = True

    if is_admin:
        if session == "all" or not session:
            return None, True
        return session, True

    # Non-admin
    req_key = await get_request_session_identifier(request)
    return req_key, False


@app.get(
    "/tracked",
    tags=["Tracking"],
    summary="List tracked objects, newest first",
)
async def tracked_list(request: Request, class_name: str = None, limit: int = 200, session: str = None):
    client_ip_filter, is_admin = await _determine_session_filter(request, session)
    objects = await list_tracked(class_name=class_name or None, limit=limit, client_ip=client_ip_filter)
    
    if is_admin:
        sessions = await list_unique_sessions()
    else:
        current_sess = await get_request_session_identifier(request)
        sessions = [current_sess] if current_sess else []
        
    return {
        "objects": objects,
        "total": len(objects),
        "client_ip": client_ip_filter or "all",
        "sessions": sessions
    }


@app.get(
    "/tracked/classes",
    tags=["Tracking"],
    summary="List class names tracked",
)
async def tracked_classes(request: Request, session: str = None):
    client_ip_filter, _ = await _determine_session_filter(request, session)
    classes = await list_classes(client_ip=client_ip_filter)
    return {"classes": classes}


@app.delete(
    "/tracked/{global_id}",
    tags=["Tracking"],
    summary="Delete all records for a tracked object",
)
async def tracked_delete(global_id: str):
    await delete_tracked(global_id)
    return {"deleted": global_id}


@app.post(
    "/tracked/search",
    tags=["Tracking"],
    summary="Search for any object by photo (ReID embedding match)",
)
async def tracked_search(
    request: Request,
    file: UploadFile = File(...),
    class_name: str = Form(None),
    threshold: float = Form(0.65),
    limit: int = Form(5),
    session: str = Form(None),
):
    """Upload a cropped photo of any object and find matching identities."""
    client_ip_filter, _ = await _determine_session_filter(request, session)
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")
    h, w = img.shape[:2]
    emb = await extract_embedding(img, [0, 0, w, h])
    if emb is None:
        raise HTTPException(500, "Re-ID embedding failed — check Triton/osnet")
    hits = await search_object_with_hits(
        emb,
        class_name=class_name or None,
        threshold=threshold,
        limit=limit,
        client_ip=client_ip_filter,
    )
    return {"results": hits, "embedding_dim": len(emb)}


@app.get(
    "/tracked/{global_id}/similar",
    tags=["Tracking"],
    summary="Find similar tracked objects by stored embedding (cross-camera Re-ID)",
)
async def tracked_similar(
    global_id: str,
    request: Request,
    threshold: float = 0.55,
    limit: int = 10,
    class_name: str = None,
    session: str = None,
):
    """Given a global_id, find other tracked objects with similar Re-ID embeddings."""
    client_ip_filter, _ = await _determine_session_filter(request, session)
    emb = await get_object_embedding(global_id)
    if emb is None:
        raise HTTPException(404, f"No embedding found for '{global_id}'")
    hits = await search_object_with_hits(
        emb,
        class_name=class_name or None,
        threshold=threshold,
        limit=limit + 1,     # +1 so we can remove the self-match
        client_ip=client_ip_filter,
    )
    hits = [h for h in hits if h["global_id"] != global_id][:limit]
    return {"global_id": global_id, "results": hits, "count": len(hits)}


# ═══════════════════════════════════════════════════════════════════
#  Model management
# ═══════════════════════════════════════════════════════════════════

@app.post(
    "/models/upload",
    tags=["Models"],
    summary="Upload .pt/.pth weights or ready .onnx → Triton load",
    response_description="status, model, type, export_imgsz, config.",
)
async def models_upload(
    file: UploadFile = File(..., description=".pt/.pth model weights or compatible YOLO-style .onnx"),
    name: Optional[str] = Form(
        None,
        description="Triton model name. If omitted, derived from filename (e.g. fall.pt → fall).",
    ),
    config: Optional[str] = Form(
        None, description="Optional JSON config overrides (max_batch_size, instance_group, …)"
    ),
    gpus: Optional[str] = Form(
        None,
        description="Shortcut: GPU index(es) for instance_group, e.g. '0' or '1' or '0,1'. "
        "See GET /gpus. Overridden by config.instance_group if both set.",
    ),
    imgsz: Optional[str] = Form(
        None,
        description="Export letterbox size (default 640). Square or height,width. Runtime may use other sizes if ONNX is dynamic.",
    ),
    dynamic: bool = Form(
        True,
        description="Export ONNX with dynamic batch/spatial dims (recommended for Triton).",
    ),
    yoloe_dynamic: bool = Form(
        False,
        description="For YOLOE .pt: export two-input ONNX (images + prompt_embedding) for text prompting. Off by default for normal YOLO uploads.",
    ),
    labels: Optional[str] = Form(
        None,
        description="Optional class labels to save immediately. Accepts newline or comma separated text.",
    ),
    overwrite: bool = Form(
        False,
        description="Replace existing model directory and reload if name already exists.",
    ),
):
    """
    Upload Ultralytics `.pt`/`.pth` weights or a compatible YOLO-style `.onnx`.

  - **name:** optional; default = filename stem (`fall.pt` → `fall`).
  - **overwrite:** `true` replaces existing model (409 if false and exists).
  - **yoloe_dynamic:** opt-in; `true` exports two-input ONNX for YOLOE text prompts.
  - **.onnx upload:** skips export, validates input/output compatibility, normalizes tensor names.
  - **imgsz:** export square size (default 640); runtime `/detect` may use other sizes.
  - **gpus:** e.g. `0` or `1` — sets `instance_group`.

  **Errors:** 400 validation, 409 duplicate, 422 export failed, 502 Triton load failed.
    """
    pt_bytes = await file.read()
    try:
        model_name = resolve_upload_name(name, file.filename)
        export_imgsz_h, export_imgsz_w = parse_imgsz(imgsz)
        if export_imgsz_h != export_imgsz_w:
            raise APIError(
                "Upload export imgsz must be square (one number). "
                "Use runtime imgsz on /detect for rectangular inference.",
                status_code=400,
            )
        export_imgsz = export_imgsz_h
    except APIError as exc:
        raise HTTPException(exc.status_code, exc.message)

    config_overrides: Optional[dict] = None
    if config:
        try:
            config_overrides = json.loads(config)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"Invalid config JSON: {exc}")

    if gpus:
        try:
            gpu_ids = [int(x.strip()) for x in gpus.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(400, "gpus must be comma-separated integers, e.g. '0' or '1'")
        if not gpu_ids:
            raise HTTPException(400, "gpus must list at least one GPU index")
        ig = [{"count": 1, "kind": "KIND_GPU", "gpus": gpu_ids}]
        try:
            validate_instance_groups(ig)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        config_overrides = config_overrides or {}
        if "instance_group" not in config_overrides:
            config_overrides["instance_group"] = ig

    try:
        result = await upload_model_file(
            model_repo=MODEL_REPO_PATH,
            triton_http_url=TRITON_HTTP_URL,
            model_name=model_name,
            model_bytes=pt_bytes,
            config_overrides=config_overrides,
            dynamic_export=dynamic,
            yoloe_dynamic_export=yoloe_dynamic,
            export_imgsz=export_imgsz,
            overwrite=overwrite,
            filename=file.filename,
        )
        _model_registry.pop(model_name, None)
        _encoders_by_model.pop(model_name, None)
        _bad_model_encoders.discard(model_name)
        _model_failures.pop(model_name, None)
        if labels:
            parsed_labels = [x.strip() for x in labels.replace("\r", "\n").replace(",", "\n").split("\n") if x.strip()]
            if parsed_labels:
                write_labels(MODEL_REPO_PATH, model_name, parsed_labels)
                result["labels_count"] = len(parsed_labels)
        return JSONResponse(result)
    except APIError as exc:
        raise HTTPException(exc.status_code, exc.message)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Upload failed for model '{model_name}'")
        raise HTTPException(500, str(exc))


@app.delete(
    "/models/{name}",
    tags=["Models"],
    summary="Delete model",
    response_description="status deleted, model name.",
)
async def models_delete(name: str):
    """Unload from Triton and remove `model_repo/{name}/`. Returns 404 if missing."""
    model_dir = os.path.join(MODEL_REPO_PATH, name)
    if not os.path.isdir(model_dir) and not is_ensemble(MODEL_REPO_PATH, name):
        raise HTTPException(404, f"Model '{name}' not found")
    try:
        ensemble_changes = []
        if not is_ensemble(MODEL_REPO_PATH, name):
            ensemble_changes = await _remove_model_from_dependent_ensembles(name)
        await delete_model(MODEL_REPO_PATH, TRITON_HTTP_URL, name)
        _model_registry.pop(name, None)
        _encoders_by_model.pop(name, None)
        _bad_model_encoders.discard(name)
        _model_failures.pop(name, None)
        return {"status": "deleted", "model": name, "ensemble_changes": ensemble_changes}
    except Exception as exc:
        raise HTTPException(500, str(exc))


class RenameModelBody(BaseModel):
    new_name: str


@app.put(
    "/models/{name}/rename",
    tags=["Models"],
    summary="Rename a model",
)
async def model_rename(name: str, body: RenameModelBody):
    """
    Rename model directory on disk, update config.pbtxt name, and hot-reload in Triton.
    Also updates dependent ensembles and active RTSP streams.
    """
    new_name = body.new_name.strip()
    if not new_name:
        raise HTTPException(400, "New model name cannot be empty.")
    if not re.match(r"^[a-zA-Z0-9_\-]+$", new_name):
        raise HTTPException(400, "New model name can only contain letters, numbers, underscores, and hyphens.")

    if new_name == name:
        return {"status": "renamed", "old_name": name, "new_name": new_name}

    old_dir = os.path.join(MODEL_REPO_PATH, name)
    new_dir = os.path.join(MODEL_REPO_PATH, new_name)

    if not os.path.isdir(old_dir):
        raise HTTPException(404, f"Model '{name}' not found on disk.")
    if os.path.exists(new_dir):
        raise HTTPException(400, f"A model named '{new_name}' already exists.")

    affected = await _drain_model_from_streams(name)

    try:
        if is_ensemble(MODEL_REPO_PATH, name):
            ens_kind = get_ensemble_kind(MODEL_REPO_PATH, name)
            if ens_kind == "native":
                try:
                    await triton_unload_model(TRITON_HTTP_URL, name)
                except Exception:
                    pass
            shutil.move(old_dir, new_dir)
            update_model_config(MODEL_REPO_PATH, new_name, {"name": new_name}, allow_protected=True)
            if ens_kind == "native":
                await triton_load_model(TRITON_HTTP_URL, new_name)
            ensemble_changes = []
        else:
            try:
                await triton_unload_model(TRITON_HTTP_URL, name)
            except Exception as exc:
                logger.warning(f"Unload of '{name}' before rename failed: {exc}")

            shutil.move(old_dir, new_dir)
            update_model_config(MODEL_REPO_PATH, new_name, {"name": new_name}, allow_protected=True)

            meta_path = os.path.join(new_dir, MODEL_META_FILE)
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        meta_data = json.load(f)
                    meta_data["model_name"] = new_name
                    with open(meta_path, "w") as f:
                        json.dump(meta_data, f, indent=2)
                except Exception:
                    pass

            ensemble_changes = await _rename_model_in_dependent_ensembles(name, new_name)
            await triton_load_model(TRITON_HTTP_URL, new_name)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Failed to rename model '{name}' to '{new_name}': {exc}")
        raise HTTPException(500, f"Rename failed: {exc}")
    finally:
        _restore_model_to_streams(new_name, affected)

    _model_registry.pop(name, None)
    _encoders_by_model.pop(name, None)
    _bad_model_encoders.discard(name)
    _model_failures.pop(name, None)

    return {
        "status": "renamed",
        "old_name": name,
        "new_name": new_name,
        "ensemble_changes": ensemble_changes,
    }


@app.get(
    "/models",
    tags=["Models"],
    summary="List Triton models (single vs ensemble)",
    response_description="models, single_models, ensemble_models with kind and platform.",
)
async def models_list(request: Request):
    """
    List models known to Triton, split for clients:

  - **models** — all entries (backward compatible)
  - **single_models** — ONNX / YOLO / YOLOE (not ensemble)
  - **ensemble_models** — `platform: ensemble` pipelines only

  Each item includes `kind` (`single` | `ensemble`) and `platform` when known.
    """
    try:
        await _ensure_triton_has_disk_models()
        index = await list_models(TRITON_HTTP_URL)
        res = split_models_by_kind(MODEL_REPO_PATH, index)
        
        info = getattr(request.state, "api_key_info", None)
        if info:
            allowed = info.get("allowed_models", ["*"])
            scopes = info.get("scopes", [])
            if "*" not in allowed and "admin" not in scopes:
                res["models"] = [m for m in res.get("models", []) if m.get("name") in allowed]
                res["single_models"] = [m for m in res.get("single_models", []) if m.get("name") in allowed]
                res["ensemble_models"] = [m for m in res.get("ensemble_models", []) if m.get("name") in allowed]
        return res
    except Exception as exc:
        raise HTTPException(503, str(exc))


@app.get(
    "/models/{name}/info",
    tags=["Models"],
    summary="Get one model details",
)
async def model_info_get(name: str):
    """Return disk, Triton, config, labels, and detected model type for one model."""
    if not _model_exists_on_disk(name):
        raise HTTPException(404, f"Model '{name}' not found")

    info = _model_info(name)
    index = await _triton_index_by_name()
    triton_entry = index.get(name)
    ens_kind = get_ensemble_kind(MODEL_REPO_PATH, name)
    kind = "ensemble" if ens_kind else "single"

    config: dict[str, Any] | None = None
    try:
        config = {
            k: v
            for k, v in read_model_config(MODEL_REPO_PATH, name).items()
            if not k.startswith("_")
        }
    except FileNotFoundError:
        config = None

    return {
        "name": name,
        "kind": kind,
        "type": info.get("type"),
        "task": info.get("task"),
        "source_format": info.get("source_format"),
        "adapter": info.get("adapter"),
        "output0_layout": info.get("output0_layout"),
        "input_names": info.get("input_names"),
        "output_names": info.get("output_names"),
        "compatibility": info.get("compatibility"),
        "onnx_signature": info.get("onnx_signature"),
        "has_masks": info.get("has_masks", False),
        "ensemble_kind": ens_kind,
        "steps": info.get("steps") if kind == "ensemble" else None,
        "labels_count": _labels_count(name),
        "failure_state": _public_failure_state(name),
        "ready": (
            True
            if ens_kind == "hybrid"
            else bool(triton_entry and triton_entry.get("state") == "READY")
        ),
        "triton": triton_entry,
        "config": config,
        "signature": dict(_model_signature(name)),
    }


@app.post(
    "/models/{name}/reload",
    tags=["Models"],
    summary="Reload model in Triton",
)
async def model_reload(name: str):
    """Unload then load one model. Hybrid ensembles only refresh API cache."""
    if not _model_exists_on_disk(name):
        raise HTTPException(404, f"Model '{name}' not found")

    try:
        kind = get_ensemble_kind(MODEL_REPO_PATH, name)
        if kind == "hybrid":
            _model_registry.pop(name, None)
            _model_failures.pop(name, None)
            return {
                "status": "refreshed",
                "model": name,
                "kind": "hybrid",
                "note": "Hybrid ensembles are API-managed and are not loaded by Triton.",
            }

        await triton_unload_model(TRITON_HTTP_URL, name)
        await triton_load_model(TRITON_HTTP_URL, name)
        _model_registry.pop(name, None)
        _encoders_by_model.pop(name, None)
        _bad_model_encoders.discard(name)
        _model_failures.pop(name, None)
        return {"status": "reloaded", "model": name}
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post(
    "/models/{name}/refresh",
    tags=["Models"],
    summary="Refresh model metadata cache",
)
async def model_refresh(name: str):
    """Clear in-process model/encoder cache and re-read model metadata from disk."""
    if not _model_exists_on_disk(name):
        raise HTTPException(404, f"Model '{name}' not found")

    _model_registry.pop(name, None)
    _encoders_by_model.pop(name, None)
    _bad_model_encoders.discard(name)
    _model_failures.pop(name, None)
    info = _model_info(name)
    return {"status": "refreshed", "model": name, "info": info}


# ═══════════════════════════════════════════════════════════════════
#  Label management
# ═══════════════════════════════════════════════════════════════════

class LabelsBody(BaseModel):
    labels: list[str]


@app.put(
    "/models/{name}/labels",
    tags=["Labels"],
    summary="Set class labels",
)
async def labels_put(name: str, body: LabelsBody):
    """Replace `labels.json`. Index = `category_id` in detections. Not used for YOLOE prompts."""
    write_labels(MODEL_REPO_PATH, name, body.labels)
    return {"status": "ok", "count": len(body.labels)}


@app.get(
    "/models/{name}/labels",
    tags=["Labels"],
    summary="Get class labels",
)
async def labels_get(name: str):
    """Return `labels.json` for a model. 404 if file missing."""
    labels = read_labels(MODEL_REPO_PATH, name)
    if labels is None:
        raise HTTPException(404, f"No labels.json found for model '{name}'")
    return {"labels": labels}


@app.delete(
    "/models/{name}/labels",
    tags=["Labels"],
    summary="Delete class labels",
)
async def labels_delete(name: str):
    """Delete `labels.json` and clear the in-process labels cache."""
    removed = delete_labels(MODEL_REPO_PATH, name)
    return {"status": "ok", "deleted": removed}


# ═══════════════════════════════════════════════════════════════════
#  Config management
# ═══════════════════════════════════════════════════════════════════

@app.get(
    "/models/{name}/config",
    tags=["Config"],
    summary="Get Triton config (JSON)",
)
async def config_get(name: str):
    """Parsed `config.pbtxt` as JSON (input/output tensor defs omitted)."""
    try:
        cfg = read_model_config(MODEL_REPO_PATH, name)
    except FileNotFoundError:
        raise HTTPException(404, f"No config.pbtxt for model '{name}'")
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


@app.put(
    "/models/{name}/config",
    tags=["Config"],
    summary="Update Triton config + reload",
)
async def config_put(name: str, updates: dict[str, Any]):
    """
    Merge JSON into `config.pbtxt`, then unload → write → reload.

  **Editable:** `max_batch_size`, `dynamic_batching`, `instance_group`,
  `version_policy`, `model_warmup`, `ensemble_scheduling`.

  **Ignored:** `name`, `backend`, `input`, `output`.
    """
    try:
        if "instance_group" in updates:
            validate_instance_groups(updates["instance_group"])
        # Drain in-flight inference before unloading to avoid gRPC errors on streams
        affected = await _drain_model_from_streams(name)
        try:
            await triton_unload_model(TRITON_HTTP_URL, name)
            merged = update_model_config(MODEL_REPO_PATH, name, updates)
            await triton_load_model(TRITON_HTTP_URL, name)
        finally:
            _restore_model_to_streams(name, affected)
        _model_registry.pop(name, None)
        _model_failures.pop(name, None)
    except FileNotFoundError:
        raise HTTPException(404, f"Model '{name}' not found")
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

    return {
        "status": "reloaded",
        "model": name,
        "config": {k: v for k, v in merged.items() if not k.startswith("_")},
    }


# ═══════════════════════════════════════════════════════════════════
#  Instance-group management
# ═══════════════════════════════════════════════════════════════════

@app.get(
    "/models/{name}/instances",
    tags=["Config"],
    summary="Get instance_group",
)
async def instances_get(name: str):
    """Return GPU/CPU `instance_group` from config.pbtxt."""
    try:
        groups = get_instance_groups(MODEL_REPO_PATH, name)
    except FileNotFoundError:
        raise HTTPException(404, f"Model '{name}' not found")
    return {"instance_group": groups}


@app.put(
    "/models/{name}/instances",
    tags=["Config"],
    summary="Set instance_group + reload",
)
async def instances_put(name: str, groups: list[dict[str, Any]]):
    """
    Replace `instance_group` and hot-reload.

  Example: `[{"count": 1, "kind": "KIND_GPU", "gpus": [0]}]`
    """
    try:
        validate_instance_groups(groups)
        # Drain in-flight inference before unloading to avoid gRPC errors on streams
        affected = await _drain_model_from_streams(name)
        try:
            await triton_unload_model(TRITON_HTTP_URL, name)
            merged = update_instance_groups(MODEL_REPO_PATH, name, groups)
            await triton_load_model(TRITON_HTTP_URL, name)
        finally:
            _restore_model_to_streams(name, affected)
        _model_registry.pop(name, None)
        _model_failures.pop(name, None)
    except FileNotFoundError:
        raise HTTPException(404, f"Model '{name}' not found")
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

    return {
        "status": "reloaded",
        "model": name,
        "instance_group": merged.get("instance_group", []),
    }


# ═══════════════════════════════════════════════════════════════════
#  Ensemble management
# ═══════════════════════════════════════════════════════════════════

class EnsembleBody(BaseModel):
    """Create a parallel Triton ensemble."""

    name: str
    steps: list[dict[str, Any]]
    description: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "safety-stack",
                    "steps": [
                        {"model": "person", "version": -1},
                        {"model": "fire", "version": -1},
                    ],
                }
            ]
        }
    }


class EnsembleUpdateBody(BaseModel):
    """Replace models inside an existing ensemble."""

    steps: list[dict[str, Any]]
    description: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "steps": [
                        {"model": "person", "version": -1},
                        {"model": "helmet", "version": -1},
                    ],
                }
            ]
        }
    }


async def _validate_ensemble_steps(steps: list[dict[str, Any]]) -> None:
    if not steps:
        raise HTTPException(400, "Ensemble steps must not be empty")
    for idx, step in enumerate(steps):
        if not step.get("model"):
            raise HTTPException(400, f"steps[{idx}].model is required")

    repo_index = await list_models(TRITON_HTTP_URL)
    loaded_names = {m.get("name") for m in repo_index if isinstance(m, dict)}
    missing = [s["model"] for s in steps if s.get("model") not in loaded_names]
    if missing:
        raise HTTPException(400, f"Models not found in Triton: {missing}")


@app.post(
    "/ensemble/create",
    tags=["Ensemble"],
    summary="Create ensemble pipeline",
)
async def ensemble_create(body: EnsembleBody):
    """
    Build ensemble `config.pbtxt` and load in Triton.

  All `steps[].model` must already be loaded. Returns 409 if ensemble name exists.
    Each annotation in `/detect` includes `source_model` for ensembles.
    """
    try:
        await _validate_ensemble_steps(body.steps)

        try:
            ens_name = validate_model_name(body.name)
        except APIError as exc:
            raise HTTPException(exc.status_code, exc.message)

        ens_dir = os.path.join(MODEL_REPO_PATH, ens_name)
        if os.path.isdir(ens_dir):
            raise HTTPException(
                409,
                f"Ensemble '{ens_name}' already exists. Delete it first.",
            )

        analysis = analyze_ensemble_steps(MODEL_REPO_PATH, body.steps)
        try:
            created = create_ensemble(MODEL_REPO_PATH, ens_name, body.steps)
            if created["kind"] == "native":
                await triton_load_model(TRITON_HTTP_URL, ens_name)
                await _wait_triton_model_ready(ens_name)
        except Exception:
            try:
                await triton_unload_model(TRITON_HTTP_URL, ens_name)
            except Exception:
                pass
            delete_ensemble(MODEL_REPO_PATH, ens_name)
            _model_registry.pop(ens_name, None)
            raise
        _model_registry.pop(ens_name, None)
        _model_info(ens_name)
        return {
            "status": "loaded",
            "ensemble": ens_name,
            "kind": created["kind"],
            "hybrid": analysis["hybrid"],
            "note": (
                "Hybrid ensemble: API orchestrates YOLOE steps (prompt_embedding). "
                "Pass 'prompts' on /detect (YOLOE text only)."
                if created["kind"] == "hybrid"
                else None
            ),
        }

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get(
    "/ensemble/{name}/validate",
    tags=["Ensemble"],
    summary="Validate ensemble dependencies",
)
async def ensemble_validate(name: str):
    """Check whether an ensemble and all sub-models are usable."""
    if not is_ensemble(MODEL_REPO_PATH, name):
        raise HTTPException(404, f"Ensemble '{name}' not found")

    try:
        kind = get_ensemble_kind(MODEL_REPO_PATH, name)
        steps = parse_ensemble_steps(MODEL_REPO_PATH, name)
    except Exception as exc:
        return {
            "ensemble": name,
            "valid": False,
            "error": str(exc),
            "steps": [],
        }

    index = await _triton_index_by_name()
    missing_models: list[str] = []
    not_ready_models: list[str] = []
    step_rows: list[dict[str, Any]] = []
    requires_prompts = False

    for step in steps:
        submodel = step.get("model")
        exists = bool(submodel and _model_exists_on_disk(submodel))
        triton_entry = index.get(submodel) if submodel else None
        ready = bool(triton_entry and triton_entry.get("state") == "READY")
        sub_info = _model_info(submodel) if exists and submodel else {}
        step_requires_prompts = sub_info.get("type") == MODEL_TYPE_YOLOE_DYNAMIC

        if not exists and submodel:
            missing_models.append(submodel)
        if exists and not ready and submodel:
            not_ready_models.append(submodel)
        if step_requires_prompts:
            requires_prompts = True

        step_rows.append({
            **step,
            "exists": exists,
            "ready": ready,
            "type": sub_info.get("type"),
            "task": sub_info.get("task"),
            "has_masks": sub_info.get("has_masks", False),
            "requires_prompts": step_requires_prompts,
        })

    ensemble_ready = True
    if kind == "native":
        ensemble_ready = bool(index.get(name) and index[name].get("state") == "READY")

    valid = (
        not missing_models
        and not not_ready_models
        and (kind == "hybrid" or ensemble_ready)
    )

    return {
        "ensemble": name,
        "valid": valid,
        "kind": kind,
        "ensemble_ready": ensemble_ready,
        "missing_models": missing_models,
        "not_ready_models": not_ready_models,
        "requires_prompts": requires_prompts,
        "steps": step_rows,
    }


@app.get(
    "/ensemble/{name}",
    tags=["Ensemble"],
    summary="Get ensemble models",
)
async def ensemble_get(name: str):
    """Return ensemble kind and sub-model steps."""
    if not is_ensemble(MODEL_REPO_PATH, name):
        raise HTTPException(404, f"Ensemble '{name}' not found")

    try:
        kind = get_ensemble_kind(MODEL_REPO_PATH, name)
        steps = parse_ensemble_steps(MODEL_REPO_PATH, name)
        return {
            "ensemble": name,
            "kind": kind,
            "hybrid": kind == "hybrid",
            "models": [step["model"] for step in steps],
            "steps": steps,
            "count": len(steps),
        }
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.put(
    "/ensemble/{name}",
    tags=["Ensemble"],
    summary="Replace models in ensemble",
)
async def ensemble_update(name: str, body: EnsembleUpdateBody):
    """
    Replace the full step list for an existing ensemble.

  Use this to add/remove models: send the desired final `steps` array.
  All referenced sub-models must already be loaded in Triton.
    """
    if not is_ensemble(MODEL_REPO_PATH, name):
        raise HTTPException(404, f"Ensemble '{name}' not found")

    try:
        ens_name = validate_model_name(name)
    except APIError as exc:
        raise HTTPException(exc.status_code, exc.message)

    try:
        await _validate_ensemble_steps(body.steps)
        old_kind = get_ensemble_kind(MODEL_REPO_PATH, ens_name)
        ens_dir = os.path.join(MODEL_REPO_PATH, ens_name)
        backup_root = tempfile.mkdtemp(prefix=f"{ens_name}-backup-")
        backup_dir = os.path.join(backup_root, ens_name)
        if os.path.isdir(ens_dir):
            shutil.copytree(ens_dir, backup_dir)

        if old_kind == "native":
            await triton_unload_model(TRITON_HTTP_URL, ens_name)

        try:
            delete_ensemble(MODEL_REPO_PATH, ens_name)
            analysis = analyze_ensemble_steps(MODEL_REPO_PATH, body.steps)
            updated = create_ensemble(MODEL_REPO_PATH, ens_name, body.steps)
            if updated["kind"] == "native":
                await triton_load_model(TRITON_HTTP_URL, ens_name)
                await _wait_triton_model_ready(ens_name)
        except Exception:
            try:
                await triton_unload_model(TRITON_HTTP_URL, ens_name)
            except Exception:
                pass
            delete_ensemble(MODEL_REPO_PATH, ens_name)
            if os.path.isdir(backup_dir):
                shutil.copytree(backup_dir, ens_dir)
                if old_kind == "native":
                    try:
                        await triton_load_model(TRITON_HTTP_URL, ens_name)
                        await _wait_triton_model_ready(ens_name)
                    except Exception as reload_exc:
                        logger.error("Failed to reload restored ensemble %s: %s", ens_name, reload_exc)
            _model_registry.pop(ens_name, None)
            raise
        finally:
            shutil.rmtree(backup_root, ignore_errors=True)

        _model_registry.pop(ens_name, None)
        _model_info(ens_name)

        return {
            "status": "updated",
            "ensemble": ens_name,
            "kind": updated["kind"],
            "hybrid": analysis["hybrid"],
            "models": [step["model"] for step in updated["steps"]],
            "steps": updated["steps"],
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.delete(
    "/ensemble/{name}",
    tags=["Ensemble"],
    summary="Delete ensemble",
)
async def ensemble_delete(name: str):
    """Unload ensemble and remove its directory from model_repo."""
    try:
        await triton_unload_model(TRITON_HTTP_URL, name)
        delete_ensemble(MODEL_REPO_PATH, name)
        _model_registry.pop(name, None)
        return {"status": "deleted", "ensemble": name}
    except Exception as exc:
        raise HTTPException(500, str(exc))


# ── NVR Tracking Endpoints (Week 8) ──────────────────────────────────────────

async def process_nvr_event(
    filename: str,
    camera_id: str,
    timestamp: str,
    local_track_id: str,
    class_name: str,
    client_ip: str,
    video_filename: str = None,
    video_offset_seconds: float = None,
    camera_name: str = None,
):
    """Asynchronous background task to process crop image, perform Re-ID, and log to Qdrant."""
    if not camera_name and 'managed_streams' in globals():
        st = managed_streams.get(camera_id)
        if st and getattr(st, 'name', None):
            camera_name = st.name
    import cv2
    import uuid
    import os
    from datetime import datetime, timezone
    from reid_client import extract_embedding
    from database import search_object_with_meta, add_object_event

    MIN_TRAVEL_SECONDS = float(os.getenv("MIN_TRAVEL_SECONDS", "5.0"))

    abs_path = os.path.join(_EVENTS_DIR, filename)
    img = cv2.imread(abs_path)
    if img is None:
        logger.error(f"[NVR Background] Cannot read crop image: {abs_path}")
        return

    # 1. Extract Re-ID embedding using Triton OSNet
    h, w, _ = img.shape
    emb = await extract_embedding(img, [0, 0, w, h])
    if emb is None:
        logger.error(f"[NVR Background] OSNet embedding extraction failed for {abs_path}")
        return

    # 2. Search Qdrant for existing entity (threshold = 0.78, person only)
    #    + Spatio-temporal travel-speed filter: reject if different camera < MIN_TRAVEL_SECONDS ago
    global_id = None
    if class_name == "person":
        match = await search_object_with_meta(emb, class_name=class_name, threshold=0.78, client_ip=client_ip)
        if match:
            match_camera = match.get("camera_id")
            match_ts_str = match.get("timestamp")
            reject = False
            if match_camera and match_camera != camera_id and match_ts_str:
                try:
                    match_ts = datetime.fromisoformat(match_ts_str.replace("Z", "+00:00"))
                    now_ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00")) if timestamp else datetime.now(timezone.utc)
                    dt = abs((now_ts - match_ts).total_seconds())
                    if dt < MIN_TRAVEL_SECONDS:
                        reject = True
                        logger.info(
                            f"[Re-ID Spatio-Temporal REJECT] Match {match['global_id']} seen on "
                            f"{match_camera} only {dt:.1f}s ago — impossible transit to {camera_id}. "
                            f"Treating as new entity."
                        )
                except Exception as st_err:
                    logger.warning(f"[Re-ID ST filter] timestamp parse error: {st_err}")
            if not reject:
                global_id = match["global_id"]

    if global_id:
        logger.info(f"[NVR Re-ID Match] Local ID {local_track_id} on {camera_id} matched global ID {global_id}")
    else:
        prefix = class_name[:3].upper() if (class_name and class_name != "unknown") else "OBJ"
        random_suffix = uuid.uuid4().hex[:6].upper()
        global_id = f"{prefix}-{random_suffix}"
        logger.info(f"[NVR Re-ID New] Local ID {local_track_id} on {camera_id} is new entity. Assigned {global_id}")

    # 3. Add to Qdrant with the client-specified timestamp and metadata
    image_relative_path = f"/events_images/{filename}"
    await add_object_event(
        global_id=global_id,
        embedding=emb,
        class_name=class_name,
        camera_id=camera_id,
        image_path=image_relative_path,
        client_ip=client_ip,
        timestamp=timestamp,
        video_filename=video_filename,
        video_offset_seconds=video_offset_seconds,
        camera_name=camera_name,
    )


@app.post("/api/v1/tracking/event", tags=["NVR Tracking"])
async def tracking_event(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    camera_id: str = Form(...),
    timestamp: str = Form(...),
    local_track_id: str = Form(...),
    class_name: str = Form("unknown"),
    video_filename: str = Form(None),
    video_offset_seconds: float = Form(None),
    camera_name: str = Form(None),
):
    """Client posts a local crop event (YOLO detection + local track). The API enqueues Re-ID in background."""
    client_ip = await get_request_session_identifier(request)

    import uuid as _uuid_mod, os as _os_mod
    ext = _os_mod.path.splitext(file.filename or "")[1] or ".jpg"
    filename = f"{_uuid_mod.uuid4().hex}{ext}"
    abs_path = _os_mod.path.join(_EVENTS_DIR, filename)

    try:
        content = await file.read()
        with open(abs_path, "wb") as f:
            f.write(content)
    except Exception as exc:
        raise HTTPException(500, f"Failed to save uploaded crop: {exc}")

    background_tasks.add_task(
        process_nvr_event,
        filename,
        camera_id,
        timestamp,
        local_track_id,
        class_name,
        client_ip,
        video_filename,
        video_offset_seconds,
        camera_name,
    )

    return {
        "status": "queued",
        "image_path": f"/events_images/{filename}",
        "camera_id": camera_id,
        "timestamp": timestamp,
        "local_track_id": local_track_id,
        "video_filename": video_filename,
        "video_offset_seconds": video_offset_seconds,
    }



async def _build_detailed_trajectory(global_id: str) -> dict:
    """Build investigation modal payload for a global_id.
    Returns a single canonical event dict (the point with the most trail data).
    The UI reads events[0] directly — no multi-event timeline.
    """
    event = await get_trajectory(global_id)
    if not event:
        return {
            "global_id": global_id,
            "class_name": "unknown",
            "first_seen": None,
            "last_seen": None,
            "dwell_seconds": 0,
            "total_events": 0,
            "cameras": [],
            "key_frames": {"main": None, "start": None, "end": None},
            "events": [],
        }

    first_seen = event.get("timestamp")
    last_seen  = event.get("last_seen") or first_seen

    dwell_sec = 0.0
    try:
        if first_seen and last_seen:
            from datetime import datetime as _dt
            t1 = _dt.fromisoformat(first_seen.replace('Z', '+00:00')).timestamp()
            t2 = _dt.fromisoformat(last_seen.replace('Z', '+00:00')).timestamp()
            dwell_sec = max(0.0, round(t2 - t1, 2))
    except Exception:
        pass

    cam_id = event.get("camera_id")
    cam = {
        "camera_id": cam_id,
        "camera_name": event.get("camera_name") or cam_id,
        "first_seen": first_seen,
        "last_seen": last_seen,
        "event_count": 1,
    }

    return {
        "global_id": global_id,
        "class_name": event.get("class_name", "unknown"),
        "first_seen": first_seen,
        "last_seen": last_seen,
        "dwell_seconds": dwell_sec,
        "total_events": 1,
        "cameras": [cam],
        "key_frames": {"main": event, "start": event, "end": event},
        "events": [event],
    }


@app.get("/api/v1/tracking/trajectories/{global_id}", tags=["NVR Tracking"])
async def get_trajectories_endpoint(global_id: str):
    """Retrieve all chronological occurrences, camera metadata and 3 keyframes (start, main, end) for a given global ID."""
    return await _build_detailed_trajectory(global_id)


@app.get("/tracked/{global_id}/trajectory", tags=["Tracking"])
async def get_tracked_trajectory_endpoint(global_id: str):
    """Retrieve full trajectory with camera metadata and 3 keyframes (alias endpoint)."""
    return await _build_detailed_trajectory(global_id)


@app.post("/api/v1/tracking/search-by-image", tags=["NVR Tracking"])
async def search_by_image_endpoint(
    request: Request,
    file: UploadFile = File(...),
    class_name: Optional[str] = Form(None),
    limit: int = Form(50),
    threshold: float = Form(0.60),
    session: Optional[str] = Form(None),
):
    """Upload a query photo to retrieve similar trajectories from the vector DB."""
    client_ip_filter, _ = await _determine_session_filter(request, session)

    # Read image content
    import cv2
    import numpy as np
    try:
        content = await file.read()
        arr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as exc:
        raise HTTPException(400, f"Cannot parse uploaded image: {exc}")

    if img is None:
        raise HTTPException(400, "Invalid image data")

    # Extract Re-ID embedding
    h, w, _ = img.shape
    emb = await extract_embedding(img, [0, 0, w, h])
    if emb is None:
        raise HTTPException(500, "Failed to extract features via Triton OSNet")

    # Query Qdrant
    hits = await search_object_with_hits(
        embedding=emb,
        class_name=class_name or None,
        threshold=threshold,
        limit=limit,
        client_ip=client_ip_filter,
    )

    return {"hits": hits}


# =============================================================================
# ADMIN ACCOUNT MANAGEMENT ENDPOINTS
# =============================================================================
class AccountCreateRequest(BaseModel):
    username: str
    password: str

class AccountUpdateRequest(BaseModel):
    password: str

@app.get("/api/v1/admin/accounts", tags=["Admin Account Management"])
async def get_accounts(session: str = Depends(require_admin)):
    # 1. Virtual default admin
    accounts = [{"username": "admin", "role": "admin", "is_default": True}]
    # 2. Created database accounts
    rows = await db_fetch_all("SELECT username, role FROM admin_accounts")
    for r in rows:
        accounts.append({
            "username": r["username"],
            "role": r["role"],
            "is_default": False
        })
    return {"accounts": accounts}

@app.post("/api/v1/admin/accounts", tags=["Admin Account Management"])
async def create_account(payload: AccountCreateRequest, session: str = Depends(require_admin)):
    username = payload.username.strip()
    if not username:
        raise HTTPException(400, "Username cannot be empty.")
    if username == "admin":
        raise HTTPException(400, "The default admin account already exists.")
        
    password_hash = hashlib.sha256(payload.password.encode()).hexdigest()
    try:
        await db_execute(
            "INSERT INTO admin_accounts (username, password_hash) VALUES (?, ?)",
            (username, password_hash)
        )
    except sqlite3.IntegrityError:
        raise HTTPException(400, "Username already exists.")
    return {"status": "ok", "message": "Account created successfully."}

@app.put("/api/v1/admin/accounts/{username}", tags=["Admin Account Management"])
async def update_account_password(username: str, payload: AccountUpdateRequest, session: str = Depends(require_admin)):
    if username == "admin":
        raise HTTPException(400, "The default admin password can only be changed via the ADMIN_PASSWORD environment variable.")
        
    password_hash = hashlib.sha256(payload.password.encode()).hexdigest()
    await db_execute(
        "UPDATE admin_accounts SET password_hash = ? WHERE username = ?",
        (password_hash, username)
    )
    return {"status": "ok", "message": "Password updated successfully."}

@app.delete("/api/v1/admin/accounts/{username}", tags=["Admin Account Management"])
async def delete_account(username: str, session: str = Depends(require_admin)):
    if username == "admin":
        raise HTTPException(400, "The default admin account cannot be deleted.")
        
    await db_execute("DELETE FROM admin_accounts WHERE username = ?", (username,))
    return {"status": "ok", "message": "Account deleted successfully."}

# =============================================================================
# API KEY REVEAL ENDPOINT
# =============================================================================
@app.get("/api/v1/admin/keys/{key_id}/reveal", tags=["Admin Key Management"])
async def reveal_key(key_id: int, session: str = Depends(require_admin)):
    row = await db_fetch_one("SELECT raw_key FROM api_keys WHERE id = ?", (key_id,))
    if not row:
        raise HTTPException(404, "API key not found.")
    return {"raw_key": row["raw_key"] or "N/A"}


