"""
postprocess.py — YOLO output → COCO-format JSON.

Handles three output layouts that ultralytics ONNX exports produce:

  Detection:     outputs[0]  [1, 4+nc,     8400]
  Segmentation:  outputs[0]  [1, 4+nc+32,  8400]
                 outputs[1]  [1, 32,       160, 160]
  YOLOE dynamic: same shapes but nc = number of text-prompt classes

  In all cases output[0] first dim after batch is (4 + classes [+ 32]):
    columns 0-3  : cx, cy, w, h  (model-input pixel space)
    columns 4 .. 4+nc-1 : per-class confidence
    columns 4+nc .. end  : mask coefficients (seg only)

Coord convention returned: COCO [x, y, w, h] in original image pixels.
Masks returned: pycocotools RLE  ({"counts": str, "size": [H, W]}).
"""
from __future__ import annotations

import numpy as np
import cv2
import os
from typing import Optional


def decode_coco_rle_mask(rle: dict) -> Optional[np.ndarray]:
    """Decode COCO RLE ``{counts, size}`` to a boolean mask ``[H, W]``."""
    try:
        from pycocotools import mask as mask_utils  # type: ignore
    except ImportError:
        return None

    counts = rle.get("counts")
    size = rle.get("size")
    if not counts or not size or len(size) != 2:
        return None

    rle_obj: dict = {"size": list(size)}
    if isinstance(counts, str):
        rle_obj["counts"] = counts.encode("utf-8")
    else:
        rle_obj["counts"] = counts

    try:
        mask = mask_utils.decode(rle_obj)
    except Exception:
        return None

    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(bool)


# ──────────────────────────── helpers ────────────────────────────

def _xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """cx,cy,w,h → x1,y1,x2,y2  (in-place safe)."""
    y = np.empty_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


_NMS_BACKEND = os.getenv("NMS_BACKEND", "auto").strip().lower()
_NMS_RUNTIME_BACKEND: Optional[str] = None
_NMS_CUDA_USABLE: Optional[bool] = None
_TV_NMS = None
_TORCH = None


def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Portable NumPy NMS. boxes: [N,4] xyxy float. Returns kept indices."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        iou = (np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)) / (
            areas[i] + areas[order[1:]] - np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1) + 1e-9
        )
        order = order[np.where(iou <= iou_thr)[0] + 1]
    return np.array(keep, dtype=np.int64)


def _nms_torchvision_cuda(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thr: float,
) -> Optional[np.ndarray]:
    global _NMS_CUDA_USABLE, _TV_NMS, _TORCH
    if _NMS_CUDA_USABLE is False:
        return None
    try:
        if _TORCH is None or _TV_NMS is None:
            import torch
            from torchvision.ops import nms as tv_nms

            _TORCH = torch
            _TV_NMS = tv_nms

        if not _TORCH.cuda.is_available():
            _NMS_CUDA_USABLE = False
            return None
        _NMS_CUDA_USABLE = True
        boxes_t = _TORCH.as_tensor(boxes, dtype=_TORCH.float32, device="cuda")
        scores_t = _TORCH.as_tensor(scores, dtype=_TORCH.float32, device="cuda")
        keep = _TV_NMS(boxes_t, scores_t, float(iou_thr))
        return keep.detach().cpu().numpy().astype(np.int64)
    except Exception:
        _NMS_CUDA_USABLE = False
        return None


def _nms_opencv(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thr: float,
) -> Optional[np.ndarray]:
    try:
        if len(boxes) == 0:
            return np.empty((0,), dtype=np.int64)
        xywh = boxes.astype(np.float32).copy()
        xywh[:, 2] = np.maximum(0.0, xywh[:, 2] - xywh[:, 0])
        xywh[:, 3] = np.maximum(0.0, xywh[:, 3] - xywh[:, 1])
        kept = cv2.dnn.NMSBoxes(
            bboxes=xywh.tolist(),
            scores=scores.astype(float).tolist(),
            score_threshold=0.0,
            nms_threshold=float(iou_thr),
        )
        if kept is None or len(kept) == 0:
            return np.empty((0,), dtype=np.int64)
        return np.array(kept, dtype=np.int64).reshape(-1)
    except Exception:
        return None


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """
    Auto NMS backend:
      1. torchvision CUDA when NVIDIA/CUDA is available
      2. OpenCV CPU on ordinary machines
      3. NumPy fallback everywhere
    """
    global _NMS_RUNTIME_BACKEND

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if boxes.size == 0 or scores.size == 0:
        return np.empty((0,), dtype=np.int64)

    backend = _NMS_BACKEND
    if backend in ("auto", "cuda", "torchvision"):
        keep = _nms_torchvision_cuda(boxes, scores, iou_thr)
        if keep is not None:
            _NMS_RUNTIME_BACKEND = "torchvision_cuda"
            return keep
        if backend in ("cuda", "torchvision"):
            _NMS_RUNTIME_BACKEND = "numpy"
            return _nms_numpy(boxes, scores, iou_thr)

    if backend in ("auto", "opencv", "cv2"):
        keep = _nms_opencv(boxes, scores, iou_thr)
        if keep is not None:
            _NMS_RUNTIME_BACKEND = "opencv"
            return keep

    _NMS_RUNTIME_BACKEND = "numpy"
    return _nms_numpy(boxes, scores, iou_thr)


def nms_backend() -> str:
    """Return currently selected NMS backend after at least one `_nms()` call."""
    return _NMS_RUNTIME_BACKEND or _NMS_BACKEND or "auto"


def _remap(boxes_xyxy: np.ndarray, meta: dict) -> np.ndarray:
    """
    Map boxes from model-input space (640×640 with letterbox padding)
    back to original image pixel coordinates.
    """
    pad_left, pad_top = meta["pad"]
    scale = meta["scale"]
    orig_h, orig_w = meta["orig_shape"]

    b = boxes_xyxy.astype(np.float32).copy()
    b[:, [0, 2]] = (b[:, [0, 2]] - pad_left) / scale
    b[:, [1, 3]] = (b[:, [1, 3]] - pad_top) / scale
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, orig_w)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, orig_h)
    return b


# ─────────────────────────── mask decode ─────────────────────────

def _crop_masks(masks: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
    """Zero pixels outside each xyxy box.  masks [N,H,W], boxes [N,4] in same space."""
    n, h, w = masks.shape
    out = masks.copy()
    for i in range(n):
        x1, y1, x2, y2 = boxes_xyxy[i]
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(w, x2)), int(min(h, y2))
        out[i, :y1] = 0
        out[i, y2:] = 0
        out[i, :, :x1] = 0
        out[i, :, x2:] = 0
    return out


def _scale_masks_to_orig(masks_lb: np.ndarray, meta: dict) -> np.ndarray:
    """
    Strip letterbox padding and resize masks from model input size to original image.
    Matches ultralytics.utils.ops.scale_masks (padding=True).
    """
    inp_h, inp_w = meta["input_size"]
    orig_h, orig_w = meta["orig_shape"]
    pad_left, pad_top = meta["pad"]
    scale = meta["scale"]

    nh_s = int(round(orig_h * scale))
    nw_s = int(round(orig_w * scale))
    pad_right = inp_w - nw_s - pad_left
    pad_bottom = inp_h - nh_s - pad_top

    top, left = pad_top, pad_left
    bottom = inp_h - pad_bottom
    right = inp_w - pad_right

    scaled: list[np.ndarray] = []
    for m in masks_lb:
        crop = m[top:bottom, left:right]
        scaled.append(
            cv2.resize(crop.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        )
    return np.stack(scaled, axis=0)


def _decode_masks(
    coeffs: np.ndarray,              # [N, 32]
    proto: np.ndarray,               # [32, mh, mw]
    boxes_xyxy_model: np.ndarray,    # [N, 4] xyxy in letterboxed model-input space
    meta: dict,
) -> list[Optional[dict]]:
    """
    Decode mask coefficients + proto → COCO RLE in original image coordinates.

    Pipeline mirrors ultralytics process_mask + scale_masks (letterbox-aware).
    """
    try:
        from pycocotools import mask as mask_utils  # type: ignore
    except ImportError:
        return [None] * len(coeffs)

    if len(coeffs) == 0:
        return []

    inp_h, inp_w = meta["input_size"]
    if proto.ndim == 4:
        proto = proto[0]
    c, mh, mw = proto.shape

    # Proto-space masks: sigmoid(coeffs @ proto) → [N, mh, mw]
    raw = 1.0 / (1.0 + np.exp(-np.einsum("nc,chw->nhw", coeffs, proto)))

    wr, hr = mw / inp_w, mh / inp_h
    boxes_proto = boxes_xyxy_model * np.array([wr, hr, wr, hr], dtype=np.float32)
    raw = _crop_masks(raw, boxes_proto)

    # Upsample to model input resolution, then un-letterbox to original size
    masks_lb = np.stack(
        [cv2.resize(raw[i], (inp_w, inp_h), interpolation=cv2.INTER_LINEAR) for i in range(len(raw))],
        axis=0,
    )
    masks_orig = _scale_masks_to_orig(masks_lb, meta)

    orig_h, orig_w = meta["orig_shape"]
    rles: list[Optional[dict]] = []
    for m in masks_orig:
        binary = (m > 0.5).astype(np.uint8)
        rle = mask_utils.encode(np.asfortranarray(binary))
        rle["counts"] = rle["counts"].decode("utf-8")
        rle["size"] = [orig_h, orig_w]
        rles.append(rle)
    return rles


# ──────────────────────────── main API ───────────────────────────

def postprocess(
    outputs: list[np.ndarray],
    meta: dict,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    class_names: Optional[list[str]] = None,
    image_id: int = 0,
    has_masks: bool = False,
    output_layout: Optional[str] = None,
) -> dict:
    """
    Convert raw Triton outputs to COCO instance-annotation format.

    Args:
        outputs        : list of numpy arrays from Triton
                         [0] → predictions  (batch dim already present)
                         [1] → proto masks  (seg models only)
        meta           : dict returned by preprocess()
        conf_threshold : minimum class confidence to keep
        iou_threshold  : NMS IoU threshold
        class_names    : list of class name strings; index = class id
        image_id       : int to embed in COCO result
        has_masks      : True for segmentation models

    Returns:
        {
          "image_id": int,
          "image_shape": [H, W],
          "annotations": [
            {
              "id": int, "category_id": int, "category_name": str,
              "score": float,
              "bbox": [x, y, w, h],   # COCO pixel coords
              "area": float,
              "segmentation": {counts, size} | null,
              "iscrowd": 0
            }, ...
          ]
        }
    """
    orig_h, orig_w = meta["orig_shape"]
    empty = {"image_id": image_id, "image_shape": [orig_h, orig_w], "annotations": []}

    raw = _normalize_yolo_raw(outputs[0], output_layout)
    if raw is None:
        return empty

    n_boxes = raw.shape[0]
    if n_boxes == 0:
        return empty

    boxes_xywh = raw[:, :4]
    rest = raw[:, 4:]

    # Separate mask coefficients (last 32 cols) from class scores
    proto: Optional[np.ndarray] = None
    coeffs_all: Optional[np.ndarray] = None
    if has_masks and len(outputs) > 1 and rest.shape[1] > 32:
        proto = outputs[1][0]          # [32, mh, mw]
        coeffs_all = rest[:, -32:]
        rest = rest[:, :-32]           # class scores only

    class_ids = np.argmax(rest, axis=1)
    max_scores = rest[np.arange(n_boxes), class_ids]

    # Confidence filter
    keep_mask = max_scores >= conf_threshold
    if not keep_mask.any():
        return empty

    boxes_xywh = boxes_xywh[keep_mask]
    max_scores = max_scores[keep_mask]
    class_ids = class_ids[keep_mask]
    coeffs_filt = coeffs_all[keep_mask] if coeffs_all is not None else None

    boxes_xyxy = _xywh2xyxy(boxes_xywh)

    # Per-class NMS (keep boxes in model space for mask decode)
    all_boxes_model: list[np.ndarray] = []
    all_boxes_orig: list[np.ndarray] = []
    all_cls: list[int] = []
    all_scores: list[float] = []
    all_coeffs: list[np.ndarray] = []

    for cls in np.unique(class_ids):
        m = class_ids == cls
        cb, cs = boxes_xyxy[m], max_scores[m]
        keep_idx = _nms(cb, cs, iou_threshold)
        for k in keep_idx:
            all_boxes_model.append(cb[k])
            all_boxes_orig.append(_remap(cb[k : k + 1], meta)[0])
            all_cls.append(int(cls))
            all_scores.append(float(cs[k]))
            if coeffs_filt is not None:
                all_coeffs.append(coeffs_filt[m][k])

    if not all_boxes_orig:
        return empty

    # Decode masks
    rles: list = [None] * len(all_boxes_orig)
    if has_masks and proto is not None and all_coeffs:
        rles = _decode_masks(
            np.array(all_coeffs),
            proto,
            np.array(all_boxes_model, dtype=np.float32),
            meta,
        )

    annotations = []
    for idx, (box, cls_id, score, rle) in enumerate(
        zip(all_boxes_orig, all_cls, all_scores, rles)
    ):
        x1, y1, x2, y2 = box
        w = float(x2 - x1)
        h = float(y2 - y1)
        name = (
            class_names[cls_id]
            if class_names and cls_id < len(class_names)
            else str(cls_id)
        )
        annotations.append({
            "id": idx,
            "category_id": cls_id,
            "category_name": name,
            "score": round(score, 4),
            "bbox": [round(float(x1), 2), round(float(y1), 2), round(w, 2), round(h, 2)],
            "area": round(w * h, 2),
            "segmentation": rle,
            "iscrowd": 0,
        })

    return {"image_id": image_id, "image_shape": [orig_h, orig_w], "annotations": annotations}


def _default_class_names(submodel: str, num_classes: int) -> list[str]:
    """Fallback labels when labels.json is missing for a sub-model."""
    if num_classes == 1:
        return [submodel]
    return [f"{submodel}_{i}" for i in range(num_classes)]


def _num_classes_from_output(tensor: np.ndarray) -> int:
    """Infer class count from YOLO head shape [1, 4+nc, anchors] or transposed."""
    raw = _normalize_yolo_raw(tensor)
    if raw is None:
        return 1
    return max(int(raw.shape[1]) - 4, 1)


def _normalize_yolo_raw(tensor: np.ndarray, output_layout: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Normalize supported YOLO raw outputs to [boxes, features].

    Explicit layouts come from model_meta.json:
      n_c_a: [N, C, anchors]
      n_a_c: [N, anchors, C]
      c_a:   [C, anchors]
      a_c:   [anchors, C]

    Without metadata, keep the legacy heuristic for old repo models.
    """
    arr = np.asarray(tensor)
    layout = (output_layout or "auto").lower()
    try:
        if layout == "n_c_a":
            if arr.ndim != 3:
                return None
            return arr[0].T
        if layout == "n_a_c":
            if arr.ndim != 3:
                return None
            return arr[0]
        if layout == "c_a":
            if arr.ndim != 2:
                return None
            return arr.T
        if layout == "a_c":
            if arr.ndim != 2:
                return None
            return arr

        raw = arr[0] if arr.ndim == 3 else arr
        if raw.ndim != 2:
            return None
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T
        return raw
    except Exception:
        return None


def _step_has_masks(model_repo: str, step: dict) -> bool:
    if "has_masks" in step:
        return bool(step["has_masks"])
    try:
        from ensemble_manager import _submodel_type

        return bool(_submodel_type(model_repo, step["model"]).get("has_masks", False))
    except Exception:
        return False


def postprocess_ensemble(
    named_outputs: dict,
    steps: list[dict],
    meta: dict,
    model_repo: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    image_id: int = 0,
    read_labels_fn=None,
) -> dict:
    """
    Run postprocess per ensemble branch and merge into one COCO JSON payload.

    Each annotation gets ``source_model`` (sub-model name) for traceability.
    """
    if read_labels_fn is None:
        from label_manager import read_labels as read_labels_fn

    orig_h, orig_w = meta["orig_shape"]
    merged: list[dict] = []
    ann_id = 0

    for step in steps:
        out_key = step["output"]
        submodel = step["model"]
        if out_key not in named_outputs:
            continue

        arr = named_outputs[out_key]
        layout = step.get("output0_layout")
        if arr.ndim == 2 and not layout:
            arr = arr[np.newaxis, ...]

        class_names = read_labels_fn(model_repo, submodel)
        if not class_names:
            class_names = _default_class_names(submodel, _num_classes_from_output(arr))

        outputs = [arr]
        has_masks = _step_has_masks(model_repo, step)
        if has_masks:
            proto_key = step.get("mask_output") or f"{submodel}_mask"
            proto = named_outputs.get(proto_key)
            if proto is not None:
                if proto.ndim == 3:
                    proto = proto[np.newaxis, ...]
                outputs.append(proto)

        part = postprocess(
            outputs=outputs,
            meta=meta,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            class_names=class_names,
            image_id=image_id,
            has_masks=has_masks and len(outputs) > 1,
            output_layout=layout,
        )

        for ann in part["annotations"]:
            ann["id"] = ann_id
            ann["source_model"] = submodel
            ann_id += 1
            merged.append(ann)

    return {
        "image_id": image_id,
        "image_shape": [orig_h, orig_w],
        "ensemble": True,
        "annotations": merged,
    }
