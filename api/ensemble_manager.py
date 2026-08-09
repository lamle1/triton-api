"""
ensemble_manager.py — Triton ensemble + API hybrid ensemble management.

Native Triton ensemble: parallel steps sharing one `images` input (YOLO-only).

Hybrid API ensemble: when any step is YOLOE dynamic (needs prompt_embedding),
orchestration runs in the API — Triton cannot map prompt_embedding per-step.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import logging
from typing import Any

logger = logging.getLogger(__name__)

HYBRID_META_FILE = "ensemble.json"


def _submodel_type(model_repo: str, model_name: str) -> dict:
    from model_detector import MODEL_META_FILE, detect_model_type

    meta_path = os.path.join(model_repo, model_name, MODEL_META_FILE)
    onnx = os.path.join(model_repo, model_name, "1", "model.onnx")
    if os.path.isfile(onnx):
        info = detect_model_type(onnx)
        if os.path.isfile(meta_path):
            try:
                with open(meta_path) as f:
                    info = {**info, **json.load(f)}
            except Exception as exc:
                logger.warning("Could not read model metadata for %s: %s", model_name, exc)
        return info
    return {"type": "yolo", "task": "detect", "has_masks": False, "num_classes": None}


def analyze_ensemble_steps(model_repo: str, steps: list[dict]) -> dict[str, Any]:
    """
    Classify ensemble steps. Returns hybrid=True if any YOLOE dynamic sub-model.
    """
    from model_detector import MODEL_TYPE_YOLOE_DYNAMIC

    enriched: list[dict] = []
    hybrid = False
    for step in steps:
        model = step["model"]
        info = _submodel_type(model_repo, model)
        if info.get("type") == MODEL_TYPE_YOLOE_DYNAMIC:
            hybrid = True
        enriched.append({
            **step,
            "output": step.get("output") or f"{model}_out",
            "version": int(step.get("version", 1)),
            "model_type": info.get("type"),
            "has_masks": info.get("has_masks", False),
            "adapter": info.get("adapter"),
            "output0_layout": info.get("output0_layout"),
            "task": info.get("task"),
        })
    return {"hybrid": hybrid, "kind": "hybrid" if hybrid else "native", "steps": enriched}


def validate_ensemble_onnx_compatibility(model_repo: str, steps: list[dict]) -> None:
    """
    Validate all sub-model ONNX files before creating an ensemble.

    Native Triton ensembles and API hybrid ensembles both rely on the same API
    tensor contract: image input named `images`, optional prompt input named
    `prompt_embedding`, and YOLO-style output0/output1 tensors. Compatible
    ONNX files are normalized in-place; incompatible ones produce detailed
    client-facing errors.
    """
    from model_detector import ONNXCompatibilityError, validate_and_normalize_onnx

    errors: list[dict[str, str]] = []
    for idx, step in enumerate(steps):
        model = step.get("model")
        onnx_path = os.path.join(model_repo, model or "", "1", "model.onnx")
        if not model or not os.path.isfile(onnx_path):
            errors.append({
                "step": str(idx),
                "model": str(model),
                "reason": "model.onnx not found in model repository",
            })
            continue
        try:
            validate_and_normalize_onnx(onnx_path)
        except ONNXCompatibilityError as exc:
            errors.append({
                "step": str(idx),
                "model": str(model),
                "reason": str(exc),
            })
    if errors:
        detail = "; ".join(
            f"steps[{e['step']}].model='{e['model']}': {e['reason']}" for e in errors
        )
        raise ValueError(f"Ensemble ONNX compatibility check failed: {detail}")


def _output_dims_from_submodel(model_repo: str, model_name: str) -> list[int]:
    """Read output0 dims from a sub-model config.pbtxt, or use fully dynamic."""
    cfg_path = os.path.join(model_repo, model_name, "config.pbtxt")
    if not os.path.exists(cfg_path):
        return [-1, -1, -1]
    with open(cfg_path) as f:
        text = f.read()
    m = re.search(
        r'name:\s*"output0"[^}]*dims:\s*\[\s*([^\]]+)\s*\]',
        text,
        re.DOTALL,
    )
    if not m:
        return [-1, -1, -1]
    dims: list[int] = []
    for part in m.group(1).split(","):
        part = part.strip()
        if part:
            dims.append(int(part))
    return dims or [-1, -1, -1]


def generate_ensemble_config(
    name: str,
    steps: list[dict],
    model_repo: str | None = None,
) -> str:
    """Build config.pbtxt for a parallel Triton ensemble (YOLO single-input only)."""
    output_blocks: list[str] = []
    step_blocks: list[str] = []

    for step in steps:
        model = step["model"]
        version = int(step.get("version", 1))
        out_name = step.get("output") or f"{model}_out"
        dims = (
            _output_dims_from_submodel(model_repo, model)
            if model_repo
            else [-1, -1, -1]
        )
        dims_str = ", ".join(str(d) for d in dims)

        output_blocks.append(
            f"  {{\n"
            f'    name:      "{out_name}"\n'
            f"    data_type: TYPE_FP32\n"
            f"    dims: [ {dims_str} ]\n"
            f"  }}"
        )
        step_blocks.append(
            f"    {{\n"
            f'      model_name: "{model}"\n'
            f"      model_version: {version}\n"
            f'      input_map  {{ key: "images"  value: "images" }}\n'
            f'      output_map {{ key: "output0" value: "{out_name}" }}\n'
            f"    }}"
        )

    outputs_text = ",\n".join(output_blocks)
    steps_text = ",\n".join(step_blocks)

    return f"""\
name: "{name}"
platform: "ensemble"
max_batch_size: 0

input [{{
  name:      "images"
  data_type: TYPE_FP32
  dims: [-1, 3, -1, -1]
}}]

output [
{outputs_text}
]

ensemble_scheduling {{
  step [
{steps_text}
  ]
}}
"""


def _write_hybrid_meta(model_repo: str, name: str, analysis: dict) -> None:
    ensemble_dir = os.path.join(model_repo, name)
    os.makedirs(ensemble_dir, exist_ok=True)
    meta = {
        "name": name,
        "kind": "hybrid",
        "steps": analysis["steps"],
    }
    with open(os.path.join(ensemble_dir, HYBRID_META_FILE), "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Hybrid ensemble meta written: {name} ({len(analysis['steps'])} steps)")


def create_ensemble(model_repo: str, name: str, steps: list[dict]) -> dict[str, Any]:
    """
    Create native Triton ensemble or API hybrid ensemble.
    Returns { kind: "native" | "hybrid", steps: [...] }.
    """
    analysis = analyze_ensemble_steps(model_repo, steps)
    validate_ensemble_onnx_compatibility(model_repo, steps)
    if analysis["hybrid"]:
        _write_hybrid_meta(model_repo, name, analysis)
        return {"kind": "hybrid", "steps": analysis["steps"]}

    ensemble_dir = os.path.join(model_repo, name)
    os.makedirs(ensemble_dir, exist_ok=True)
    os.makedirs(os.path.join(ensemble_dir, "1"), exist_ok=True)

    config_path = os.path.join(ensemble_dir, "config.pbtxt")
    config_text = generate_ensemble_config(name, steps, model_repo=model_repo)
    with open(config_path, "w") as f:
        f.write(config_text)

    logger.info(f"Native ensemble config written: {config_path}  ({len(steps)} steps)")
    return {"kind": "native", "steps": analysis["steps"]}


def get_ensemble_kind(model_repo: str, name: str) -> str | None:
    """Return 'hybrid', 'native', or None if not an ensemble."""
    if os.path.isfile(os.path.join(model_repo, name, HYBRID_META_FILE)):
        return "hybrid"
    cfg_path = os.path.join(model_repo, name, "config.pbtxt")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            if re.search(r'platform:\s*"ensemble"', f.read()):
                return "native"
    return None


def is_ensemble(model_repo: str, name: str) -> bool:
    return get_ensemble_kind(model_repo, name) is not None


def parse_ensemble_steps(model_repo: str, ensemble_name: str) -> list[dict]:
    meta_path = os.path.join(model_repo, ensemble_name, HYBRID_META_FILE)
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            data = json.load(f)
        return data.get("steps", [])

    cfg_path = os.path.join(model_repo, ensemble_name, "config.pbtxt")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"No ensemble config for '{ensemble_name}'")

    with open(cfg_path) as f:
        text = f.read()

    steps: list[dict] = []
    pattern = re.compile(
        r"model_name:\s*\"([^\"]+)\""
        r".*?model_version:\s*(-?\d+)"
        r".*?output_map\s*\{[^}]*value:\s*\"([^\"]+)\"",
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        steps.append({
            "model": m.group(1),
            "version": int(m.group(2)),
            "output": m.group(3),
        })

    if not steps:
        raise ValueError(f"No ensemble steps found in config for '{ensemble_name}'")
    return steps


def delete_ensemble(model_repo: str, name: str) -> None:
    ensemble_dir = os.path.join(model_repo, name)
    if os.path.exists(ensemble_dir):
        shutil.rmtree(ensemble_dir)
        logger.info(f"Ensemble directory removed: {ensemble_dir}")
    else:
        logger.warning(f"delete_ensemble: {ensemble_dir} not found — nothing to remove")
