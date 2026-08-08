"""
yoloe_export.py — Export YOLOE to two-input ONNX (images + prompt_embedding).

Ultralytics default export fuses text embeddings into weights (static single input).
This module exports an unfused graph so runtime prompts come from the API encoder.
"""
from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)


class _YOLOEDynamicWrapper(torch.nn.Module):
    """ONNX trace wrapper: images + prompt_embedding → YOLOE outputs."""

    def __init__(self, emodel) -> None:
        super().__init__()
        self.emodel = emodel

    def forward(self, images: torch.Tensor, prompt_embedding: torch.Tensor):
        # prompt_embedding: [B, N, 512] — same tensor family as YOLOE.get_text_pe()
        return self.emodel.predict(images, tpe=prompt_embedding)


def _prepare_emodel_for_export(emodel) -> tuple[bool, bool]:
    """Set export flags on YOLOE head; refuse if already fused."""
    from ultralytics.nn.modules.head import YOLOEDetect, YOLOESegment

    head = emodel.model[-1]
    if getattr(head, "is_fused", False):
        from inference_utils import APIError

        raise APIError(
            "YOLOE checkpoint is already fused (static). "
            "Upload unfused .pt weights for text-prompt inference.",
            status_code=422,
        )

    is_seg = isinstance(head, (YOLOESegment,))
    for m in emodel.modules():
        if isinstance(m, (YOLOEDetect, YOLOESegment)):
            m.dynamic = True
            m.export = True
            m.format = "onnx"
        elif hasattr(m, "export"):
            m.export = True

    return is_seg, isinstance(head, YOLOEDetect)


def export_yoloe_dynamic_sync(
    pt_path: str,
    output_dir: str,
    imgsz: int = 640,
    opset: int = 17,
) -> str:
    """
    Export unfused YOLOE to ONNX with inputs:
      images            [B, 3, H, W]
      prompt_embedding  [B, N, 512]

    Returns path to written model.onnx.
    """
    from ultralytics import YOLOE

    wrapper = YOLOE(pt_path)
    emodel = wrapper.model
    emodel.eval()

    is_seg, _ = _prepare_emodel_for_export(emodel)

    head = emodel.model[-1]
    n_classes = max(int(getattr(head, "nc", 80)), 1)
    embed_dim = int(getattr(head, "embed", 512))

    wrapper_mod = _YOLOEDynamicWrapper(emodel)
    images = torch.randn(1, 3, imgsz, imgsz)
    prompt_embedding = torch.randn(1, n_classes, embed_dim)

    with torch.no_grad():
        outputs = wrapper_mod(images, prompt_embedding)

    if isinstance(outputs, torch.Tensor):
        out_tensors = [outputs]
        out_names = ["output0"]
    elif isinstance(outputs, (list, tuple)):
        out_tensors = outputs
        out_names = (
            ["output0", "output1"] if len(outputs) >= 2 else [f"output{i}" for i in range(len(outputs))]
        )
    else:
        out_tensors = [outputs]
        out_names = ["output0"]

    dynamic_axes = {
        "images": {0: "batch", 2: "height", 3: "width"},
        "prompt_embedding": {0: "batch", 1: "num_classes"},
    }
    for name in out_names:
        dynamic_axes[name] = {0: "batch"}

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "model.onnx")

    export_kwargs = dict(
        input_names=["images", "prompt_embedding"],
        output_names=out_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    try:
        torch.onnx.export(wrapper_mod, (images, prompt_embedding), out_path, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(wrapper_mod, (images, prompt_embedding), out_path, **export_kwargs)

    logger.info(f"YOLOE dynamic ONNX written: {out_path}  seg={is_seg}")
    return out_path


def is_yoloe_checkpoint(pt_path: str) -> bool:
    """True only for unfused YOLOE heads (not standard YOLO .pt loaded via YOLOE())."""
    try:
        from ultralytics import YOLOE
        from ultralytics.nn.modules.head import YOLOEDetect, YOLOESegment

        m = YOLOE(pt_path)
        head = m.model.model[-1]
        return isinstance(head, (YOLOEDetect, YOLOESegment))
    except Exception:
        return False
