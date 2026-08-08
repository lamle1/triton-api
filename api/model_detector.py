"""
model_detector.py — Auto-detect model type and task from an ONNX file.

Inspection strategy
───────────────────
1. Count and name inputs:
   • 2 inputs with one matching "prompt" / "embedding" → YOLOE dynamic
   • 1 input → YOLO or YOLOE static (treated identically for inference)

2. Count outputs:
   • 2 outputs → segmentation (output0 predictions + output1 proto masks)
   • 1 output  → detection or pose

3. Check output names / shapes for pose keywords.

Returns a dict used throughout the API:
  {
    "type":       "yoloe-dynamic" | "yolo",
    "task":       "detect" | "seg" | "pose",
    "num_classes": int | None,
    "has_masks":   bool,
  }
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Exported constants used by other modules
MODEL_TYPE_YOLOE_DYNAMIC = "yoloe-dynamic"
MODEL_TYPE_YOLO = "yolo"   # covers standard YOLO + YOLOE-static

TASK_DETECT = "detect"
TASK_SEG = "seg"
TASK_POSE = "pose"

API_IMAGE_INPUT = "images"
API_PROMPT_INPUT = "prompt_embedding"
API_OUTPUT0 = "output0"
API_OUTPUT1 = "output1"
MODEL_META_FILE = "model_meta.json"


class ONNXCompatibilityError(ValueError):
    """Raised when an ONNX file cannot be used by this API's vision runtime."""


def detect_model_type(onnx_path: str) -> dict:
    """
    Inspect an ONNX model file and return a model-info dict.

    Args:
        onnx_path : absolute path to the .onnx file

    Returns:
        {
          "type":        str   MODEL_TYPE_* constant
          "task":        str   TASK_* constant
          "num_classes": int | None
          "has_masks":   bool
        }
    """
    try:
        import onnx  # type: ignore
        model = onnx.load(onnx_path)
    except Exception as e:
        logger.warning(f"Cannot load ONNX for type detection: {e}")
        return _default_info()

    input_names = [i.name.lower() for i in model.graph.input]
    output_names = [o.name.lower() for o in model.graph.output]
    n_outputs = len(model.graph.output)

    # ── model type ────────────────────────────────────────────────
    is_yoloe_dynamic = any(
        "prompt" in n or "embedding" in n for n in input_names
    )
    model_type = MODEL_TYPE_YOLOE_DYNAMIC if is_yoloe_dynamic else MODEL_TYPE_YOLO

    # ── task detection ────────────────────────────────────────────
    # Pose: output names typically contain "kpt" or "keypoint"
    is_pose = any("kpt" in n or "keypoint" in n or "pose" in n for n in output_names)

    # Segmentation: 2 outputs AND not pose
    is_seg = (n_outputs >= 2) and not is_pose

    task = TASK_DETECT
    if is_seg:
        task = TASK_SEG
    elif is_pose:
        task = TASK_POSE

    # ── num_classes from output0 shape ────────────────────────────
    num_classes: int | None = None
    try:
        out0 = model.graph.output[0]
        shape = []
        for d in out0.type.tensor_type.shape.dim:
            v = d.dim_value
            shape.append(v if v > 0 else -1)
        # Expected: [batch, 4+nc(+32), anchors] or [batch, anchors, 4+nc(+32)]
        if len(shape) in (2, 3):
            tail = shape[1:] if len(shape) == 3 else shape
            pos = [s for s in tail if s > 0]
            feat_dim = min(pos) if len(pos) >= 2 else (pos[0] if pos else None)
            if feat_dim and feat_dim > 4:
                extra = feat_dim - 4
                if is_seg and extra > 32:
                    num_classes = extra - 32
                elif not is_seg:
                    num_classes = extra
    except Exception:
        pass

    info = {
        "type": model_type,
        "task": task,
        "num_classes": num_classes,
        "has_masks": task == TASK_SEG,
    }
    logger.info(f"Model detection result for {onnx_path}: {info}")
    return info


def validate_and_normalize_onnx(onnx_path: str) -> dict[str, Any]:
    """
    Validate an uploaded ONNX file against the API runtime contract and normalize
    compatible tensor names in-place.

    Supported contracts:
      - YOLO-like detect/seg: input FP32 NCHW image, output0 rank-3 predictions
      - YOLOE dynamic: same image input + FP32 [N, prompts, 512] prompt embedding

    Rejected deliberately:
      - classifier ONNX (rank-2 logits, e.g. ResNet/ImageNet)
      - NHWC image inputs
      - non-FP32 image/prompt tensors
      - models without a YOLO-like output0 tensor
    """
    try:
        import onnx  # type: ignore
        from onnx import TensorProto  # type: ignore
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
    except Exception as exc:
        raise ONNXCompatibilityError(f"Invalid ONNX file: {exc}") from exc

    initializers = {i.name for i in model.graph.initializer}
    graph_inputs = [i for i in model.graph.input if i.name not in initializers]
    graph_outputs = list(model.graph.output)

    if len(graph_inputs) not in (1, 2):
        raise ONNXCompatibilityError(
            f"Expected 1 image input or 2 inputs (image + prompt_embedding), got {len(graph_inputs)}"
        )
    if len(graph_outputs) not in (1, 2):
        raise ONNXCompatibilityError(
            f"Expected 1 YOLO output or 2 YOLO segmentation outputs, got {len(graph_outputs)}"
        )

    def elem_type(v) -> int | None:
        return v.type.tensor_type.elem_type

    def dims(v) -> list[int | str]:
        out: list[int | str] = []
        for d in v.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                out.append(int(d.dim_value))
            elif d.dim_param:
                out.append(str(d.dim_param))
            else:
                out.append(-1)
        return out

    def is_float(v) -> bool:
        return elem_type(v) == TensorProto.FLOAT

    image_candidates = []
    prompt_candidates = []
    for inp in graph_inputs:
        shape = dims(inp)
        lname = inp.name.lower()
        if len(shape) == 4:
            image_candidates.append((inp, shape))
        elif len(shape) == 3 or "prompt" in lname or "embed" in lname:
            prompt_candidates.append((inp, shape))

    if len(image_candidates) != 1:
        raise ONNXCompatibilityError(
            "Expected exactly one rank-4 image input. "
            f"Inputs: {[{'name': i.name, 'shape': dims(i)} for i in graph_inputs]}"
        )

    image_input, image_shape = image_candidates[0]
    if not is_float(image_input):
        raise ONNXCompatibilityError(
            f"Image input '{image_input.name}' must be FP32, got ONNX elem_type={elem_type(image_input)}"
        )
    if not _dim_is_dynamic_or(image_shape[1], 3):
        raise ONNXCompatibilityError(
            f"Image input '{image_input.name}' must be NCHW with channel dimension 3. "
            f"Got shape {image_shape}. NHWC/common classifier inputs are not supported by /detect."
        )

    prompt_input = None
    if len(graph_inputs) == 2:
        prompt_candidates = [(i, s) for i, s in prompt_candidates if i.name != image_input.name]
        if len(prompt_candidates) != 1:
            raise ONNXCompatibilityError(
                "Two-input models must have one rank-3 prompt embedding input "
                "[batch, prompt_count, 512]."
            )
        prompt_input, prompt_shape = prompt_candidates[0]
        if not is_float(prompt_input):
            raise ONNXCompatibilityError(
                f"Prompt input '{prompt_input.name}' must be FP32, got ONNX elem_type={elem_type(prompt_input)}"
            )
        if len(prompt_shape) != 3 or not _dim_is_dynamic_or(prompt_shape[2], 512):
            raise ONNXCompatibilityError(
                f"Prompt input '{prompt_input.name}' must be [N, prompts, 512], got {prompt_shape}"
            )

    out0_shape = dims(graph_outputs[0])
    layout = _detect_yolo_raw_layout(out0_shape)
    if layout is None:
        raise ONNXCompatibilityError(
            f"output0 must be YOLO raw predictions shaped [N, C, anchors], [N, anchors, C], "
            f"[C, anchors], or [anchors, C]. "
            f"Got output '{graph_outputs[0].name}' shape {out0_shape}. "
            "Classifier ONNX models are not compatible with /detect."
        )
    if not is_float(graph_outputs[0]):
        raise ONNXCompatibilityError(
            f"output0 must be FP32, got ONNX elem_type={elem_type(graph_outputs[0])}"
        )

    if len(graph_outputs) == 2:
        out1_shape = dims(graph_outputs[1])
        if len(out1_shape) != 4:
            raise ONNXCompatibilityError(
                f"Segmentation output1 must be rank-4 mask prototypes [N, 32, H, W], got {out1_shape}"
            )
        if not is_float(graph_outputs[1]):
            raise ONNXCompatibilityError(
                f"output1 must be FP32, got ONNX elem_type={elem_type(graph_outputs[1])}"
            )

    _free_target_name(model, image_input.name, API_IMAGE_INPUT)
    if prompt_input is not None:
        _free_target_name(model, prompt_input.name, API_PROMPT_INPUT)
    _free_target_name(model, graph_outputs[0].name, API_OUTPUT0)
    if len(graph_outputs) == 2:
        _free_target_name(model, graph_outputs[1].name, API_OUTPUT1)

    rename_value(model, image_input.name, API_IMAGE_INPUT)
    if prompt_input is not None:
        rename_value(model, prompt_input.name, API_PROMPT_INPUT)
    rename_value(model, graph_outputs[0].name, API_OUTPUT0)
    if len(graph_outputs) == 2:
        rename_value(model, graph_outputs[1].name, API_OUTPUT1)

    try:
        onnx.checker.check_model(model)
        onnx.save(model, onnx_path)
    except Exception as exc:
        raise ONNXCompatibilityError(f"ONNX name normalization failed: {exc}") from exc

    return {
        "input_names": [i.name for i in model.graph.input if i.name not in initializers],
        "output_names": [o.name for o in model.graph.output],
        "image_shape": image_shape,
        "image_dims": _triton_dims_from_shape(image_shape),
        "output0_shape": out0_shape,
        "output0_layout": layout,
        "adapter": "yolo_raw",
        "output0_dims": _triton_dims_from_layout(out0_shape, layout),
        "output1_dims": _triton_dims_from_shape(dims(graph_outputs[1])) if len(graph_outputs) == 2 else None,
        "normalized": True,
    }


def rename_value(model, old: str, new: str) -> None:
    if old == new:
        return
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if vi.name == old:
            vi.name = new
    for node in model.graph.node:
        for idx, name in enumerate(node.input):
            if name == old:
                node.input[idx] = new
        for idx, name in enumerate(node.output):
            if name == old:
                node.output[idx] = new


def _value_name_exists(model, name: str) -> bool:
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if vi.name == name:
            return True
    return any(name in node.input or name in node.output for node in model.graph.node)


def _free_target_name(model, old: str, target: str) -> None:
    """Avoid ONNX SSA collisions before normalizing a public API tensor name."""
    if old == target or not _value_name_exists(model, target):
        return
    rename_value(model, target, f"__api_internal_{target}")


def _dim_is_dynamic_or(value: int | str, expected: int) -> bool:
    return value == expected or value == -1 or isinstance(value, str)


def _is_known_feature_dim(value: int | str) -> bool:
    return isinstance(value, int) and 4 < value <= 4096


def _is_known_anchor_dim(value: int | str) -> bool:
    return isinstance(value, int) and value > 128


def _is_anchor_axis(value: int | str) -> bool:
    return value == -1 or isinstance(value, str) or _is_known_anchor_dim(value)


def _detect_yolo_raw_layout(shape: list[int | str]) -> str | None:
    """
    Return explicit raw YOLO layout:
      n_c_a: [N, C, anchors]
      n_a_c: [N, anchors, C]
      c_a:   [C, anchors]
      a_c:   [anchors, C]
    """
    if len(shape) == 3:
        _, d1, d2 = shape
        if _is_known_feature_dim(d1) and _is_anchor_axis(d2):
            return "n_c_a"
        if _is_anchor_axis(d1) and _is_known_feature_dim(d2):
            return "n_a_c"
        if d1 == -1 and d2 == -1:
            return "n_c_a"
        if isinstance(d1, str) or isinstance(d2, str):
            return "n_c_a"
        return None
    if len(shape) == 2:
        d0, d1 = shape
        if _is_known_feature_dim(d0) and _is_anchor_axis(d1):
            return "c_a"
        if _is_anchor_axis(d0) and _is_known_feature_dim(d1):
            return "a_c"
        if d0 == -1 and d1 == -1:
            return "c_a"
        if isinstance(d0, str) or isinstance(d1, str):
            return "c_a"
    return None


def _triton_dim(v: int | str) -> int:
    return v if isinstance(v, int) and v > 0 else -1


def _triton_dims_from_shape(shape: list[int | str]) -> list[int]:
    return [_triton_dim(v) for v in shape]


def _triton_dims_from_layout(shape: list[int | str], layout: str) -> list[int]:
    return _triton_dims_from_shape(shape)


def _default_info() -> dict:
    return {
        "type": MODEL_TYPE_YOLO,
        "task": TASK_DETECT,
        "num_classes": None,
        "has_masks": False,
    }
