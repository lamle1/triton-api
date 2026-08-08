"""
preprocess.py — Image preprocessing for YOLO-family models.

Pipeline:
  1. Decode JPEG/PNG bytes → BGR ndarray
  2. Letterbox-resize to (640, 640) with grey padding
  3. Normalize to [0, 1]
  4. HWC → CHW → add batch dim → float32 [1, 3, H, W]

Returns the tensor and a metadata dict used by postprocess.py to remap
detections back to original image coordinates.
"""
import cv2
import numpy as np


def letterbox(
    img: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Resize image keeping aspect ratio; pad to new_shape with solid colour.

    Returns:
        img_padded : resized + padded image (HxWxC uint8)
        scale      : resize ratio (same for both axes)
        pad        : (pad_left, pad_top) pixels of padding added
    """
    h, w = img.shape[:2]
    nh, nw = new_shape
    scale = min(nw / w, nh / h)
    nw_s = int(round(w * scale))
    nh_s = int(round(h * scale))

    img_r = cv2.resize(img, (nw_s, nh_s), interpolation=cv2.INTER_LINEAR)

    dw = nw - nw_s
    dh = nh - nh_s
    pad_top = dh // 2
    pad_bottom = dh - pad_top
    pad_left = dw // 2
    pad_right = dw - pad_left

    img_p = cv2.copyMakeBorder(
        img_r, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=color,
    )
    return img_p, scale, (pad_left, pad_top)


def preprocess(
    image_bytes: bytes,
    input_size: tuple[int, int] = (640, 640),
) -> tuple[np.ndarray, dict]:
    """
    Preprocess raw image bytes for Triton inference.

    Args:
        image_bytes : raw JPEG/PNG bytes
        input_size  : model input resolution, default (640, 640)

    Returns:
        tensor : float32 ndarray  [1, 3, H, W]  values in [0, 1]
        meta   : dict with keys:
                   orig_shape  → (orig_h, orig_w)
                   scale       → float resize ratio
                   pad         → (pad_left, pad_top) pixels
                   input_size  → (H, W)
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image — unsupported format or corrupt data.")

    orig_h, orig_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_lb, scale, (pad_left, pad_top) = letterbox(img_rgb, new_shape=input_size)

    # Normalize → transpose → add batch dim
    tensor = (img_lb.astype(np.float32) / 255.0)   # HWC float32
    tensor = np.transpose(tensor, (2, 0, 1))        # CHW
    tensor = np.expand_dims(tensor, 0)              # 1CHW
    tensor = np.ascontiguousarray(tensor)

    meta = {
        "orig_shape": (orig_h, orig_w),
        "scale": scale,
        "pad": (pad_left, pad_top),
        "input_size": input_size,
    }
    return tensor, meta
