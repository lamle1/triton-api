# ─────────────────────────────────────────────────────────────────
# Vision Inference API Server
# FastAPI + Uvicorn  |  port 8003
# ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System dependencies
# libgl1 + libglib2.0 → OpenCV
# libgomp1            → ONNX Runtime / PyTorch OpenMP
# gcc / g++           → pycocotools C extension build
# gstreamer*          → optional low-latency RTSP backend when OpenCV supports CAP_GSTREAMER
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        gcc \
        g++ \
        git \
        pkg-config \
        libcairo2-dev \
        libgirepository-2.0-dev \
        libgirepository1.0-dev \
        gir1.2-gstreamer-1.0 \
        gir1.2-gst-plugins-base-1.0 \
        gir1.2-gst-plugins-bad-1.0 \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-libav \
        gstreamer1.0-plugins-bad-apps \
    && rm -rf /var/lib/apt/lists/*

# Print available NVIDIA GStreamer decoders at build time.
# nvh264dec = hardware H.264 decode on GTX 10xx/16xx (NVDEC).
# If this prints "no element nvh264dec" the runtime will fall back to avdec_h264 (CPU).
# To get nvh264dec on Debian/Ubuntu you may need to add the CUDA apt repo first:
#   https://developer.nvidia.com/cuda-downloads → choose deb(network)
# On the host: apt install gstreamer1.0-plugins-bad will include nvh264dec on
# an NVIDIA CUDA-capable machine. Inside this python:3.11-slim container the
# CUDA-aware gstreamer package is NOT available — nvh264dec will be absent.
# ─── Workaround: install gstreamer1.0-nice + gst-plugins-nvenc from CUDA repo
# OR switch to FROM nvcr.io/nvidia/cuda:12.x-runtime-ubuntu22.04 base image.
RUN gst-inspect-1.0 nvh264dec 2>/dev/null && echo "nvh264dec: AVAILABLE (HW decode)" \
    || echo "nvh264dec: NOT FOUND → will use avdec_h264 (CPU decode); see Dockerfile comments to enable HW decode"

# Install Python deps first (layer cache friendly)
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir PyGObject

# Copy application source & export script
COPY api/*.py ./
COPY api/admin ./admin
COPY scripts/export_osnet.py ./export_osnet.py

# ── Bake Re-ID ONNX model into the image ──────────────────────────────────
# Stored at /reid_model/osnet inside the image. On startup, copied to /model_repo/osnet if missing.
COPY models/osnet /reid_model/osnet

# ── Bake MobileCLIP prompt encoder asset & YOLOE weights into the image ─────
COPY weights/yoloe-v8s-seg.pt /app/yoloe-v8s-seg.pt
RUN python3 -c "from yoloe_assets import ensure_mobileclip_asset; ensure_mobileclip_asset()" 2>&1 \
 || echo "WARNING: MobileCLIP asset download failed — will download at runtime if needed"

# Mount-point stubs (actual dirs come from compose volumes)
RUN mkdir -p /model_repo /weights /tmp/uploads /events_images

EXPOSE 8003

# Entrypoint: copy baked Re-ID model, MobileCLIP asset, and YOLOE weights into volumes if not already present,
# then start uvicorn.
CMD ["sh", "-c", "\
  if [ -d /app/src ]; then \
    echo '[startup] Syncing source files from /app/src...'; \
    cp /app/src/*.py /app/ 2>/dev/null || true; \
    cp -r /app/src/admin /app/ 2>/dev/null || true; \
  fi; \
  if [ -d /reid_model/osnet ] && [ ! -d /model_repo/osnet/1 ]; then \
    echo '[startup] Copying baked Re-ID model to Triton model_repo...'; \
    mkdir -p /model_repo/osnet; \
    cp -r /reid_model/osnet/* /model_repo/osnet/ 2>/dev/null || true; \
  fi; \
  if [ -f /app/mobileclip_blt.ts ] && [ ! -f /weights/mobileclip_blt.ts ] && [ -d /weights ]; then \
    echo '[startup] Copying baked MobileCLIP asset to /weights...'; \
    cp /app/mobileclip_blt.ts /weights/mobileclip_blt.ts 2>/dev/null || true; \
  fi; \
  if [ -f /app/yoloe-v8s-seg.pt ] && [ ! -f /weights/yoloe-v8s-seg.pt ] && [ -d /weights ]; then \
    echo '[startup] Copying baked YOLOE weights to /weights...'; \
    cp /app/yoloe-v8s-seg.pt /weights/yoloe-v8s-seg.pt 2>/dev/null || true; \
  fi; \
  uvicorn main:app \
    --host 0.0.0.0 \
    --port ${API_PORT:-8003} \
    --workers ${UVICORN_WORKERS:-4} \
    --log-level info"]



