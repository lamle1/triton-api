"""
config_manager.py — Read, merge, and write Triton config.pbtxt files.

Uses a regex-based parser rather than the protobuf library so we don't
need the Triton .proto files at runtime.

Protected fields (never overwritten on PUT /config):
  name, backend, input, output

Editable fields:
  max_batch_size, dynamic_batching, instance_group, version_policy,
  model_warmup, ensemble_scheduling
"""
from __future__ import annotations

import os
import re
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Fields the client is NEVER allowed to overwrite
PROTECTED_FIELDS = {"name", "backend", "input", "output"}


# ──────────────────────────── parsing ────────────────────────────

def parse_config(text: str) -> dict:
    """
    Parse a config.pbtxt string into a Python dict.

    Input/output tensor blocks are preserved verbatim under the private
    keys "_input_raw" and "_output_raw" so they survive a round-trip.
    """
    cfg: dict[str, Any] = {}

    # ── scalar fields ─────────────────────────────────────────────
    m = re.search(r'^\s*name\s*:\s*"([^"]+)"', text, re.MULTILINE)
    if m:
        cfg["name"] = m.group(1)

    m = re.search(r'^\s*backend\s*:\s*"([^"]+)"', text, re.MULTILINE)
    if m:
        cfg["backend"] = m.group(1)

    m = re.search(r'^\s*platform\s*:\s*"([^"]+)"', text, re.MULTILINE)
    if m:
        cfg["platform"] = m.group(1)

    m = re.search(r'^\s*max_batch_size\s*:\s*(\d+)', text, re.MULTILINE)
    if m:
        cfg["max_batch_size"] = int(m.group(1))

    # ── dynamic_batching block ────────────────────────────────────
    db_m = re.search(r'dynamic_batching\s*\{([^}]+)\}', text, re.DOTALL)
    if db_m:
        db_text = db_m.group(1)
        db: dict[str, Any] = {}
        pbs = re.findall(r'preferred_batch_size\s*:\s*(\d+)', db_text)
        if pbs:
            db["preferred_batch_size"] = [int(x) for x in pbs]
        delay = re.search(r'max_queue_delay_microseconds\s*:\s*(\d+)', db_text)
        if delay:
            db["max_queue_delay_microseconds"] = int(delay.group(1))
        cfg["dynamic_batching"] = db

    # ── instance_group blocks ─────────────────────────────────────
    # Supports:  instance_group [ { ... } { ... } ]  (standard multi-line Triton format)
    # Uses a brace-depth extractor to avoid [^}]+ stopping at the first inner '}'.
    ig_outer = re.search(r'instance_group\s*\[([^\[]*(?:\{[^}]*\}[^\[]*)*)\]', text, re.DOTALL)
    if not ig_outer:
        # Fallback: single-block without outer brackets  instance_group { ... }
        ig_outer = re.search(r'instance_group\s*\{([^}]*)\}', text, re.DOTALL)
    if ig_outer:
        groups = []
        # Extract every { ... } block inside the outer brackets
        inner = ig_outer.group(1)
        for igt in re.finditer(r'\{([^}]*)\}', inner, re.DOTALL):
            block = igt.group(1)
            g: dict[str, Any] = {}
            cm = re.search(r'count\s*:\s*(\d+)', block)
            if cm:
                g["count"] = int(cm.group(1))
            km = re.search(r'kind\s*:\s*(KIND_\w+)', block)
            if km:
                g["kind"] = km.group(1)
            gm = re.search(r'gpus\s*:\s*\[\s*([^\]]+)\s*\]', block)
            if gm:
                g["gpus"] = [int(x.strip()) for x in gm.group(1).split(",") if x.strip()]
            if g:
                groups.append(g)
        if groups:
            cfg["instance_group"] = groups

    # ── version_policy ────────────────────────────────────────────
    vp_m = re.search(
        r'version_policy\s*\{[^}]*latest\s*\{[^}]*num_versions\s*:\s*(\d+)', text, re.DOTALL
    )
    if vp_m:
        cfg["version_policy"] = {"latest": {"num_versions": int(vp_m.group(1))}}

    # ── preserve input / output blocks verbatim ───────────────────
    # We capture the full "input { ... }" block (handles nested braces via
    # a simple depth counter — good enough for the flat Triton format).
    cfg["_input_raw"] = _extract_blocks(text, "input")
    cfg["_output_raw"] = _extract_blocks(text, "output")

    return cfg


def _extract_blocks(text: str, keyword: str) -> list[str]:
    """Extract all top-level `keyword { ... }` blocks from text."""
    results = []
    pattern = re.compile(rf'\b{keyword}\s*\{{', re.MULTILINE)
    for m in pattern.finditer(text):
        start = m.start()
        depth = 0
        i = m.end() - 1   # points at opening '{'
        while i < len(text):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    results.append(text[start : i + 1])
                    break
            i += 1
    return results


# ─────────────────────────── serialisation ───────────────────────

def _adjust_dims_for_batch_size(block_str: str, max_batch_size: int) -> str:
    """
    Adjust dims in protobuf input/output blocks based on max_batch_size.
    When max_batch_size > 0, Triton automatically adds the batch dim, so
    the leading dynamic batch dim (-1) MUST be omitted from dims: [...].
    When max_batch_size == 0, the batch dim (-1) MUST be included in dims: [...].
    """
    def _fix_dims(match: re.Match) -> str:
        dims_content = match.group(1).strip()
        dims = [d.strip() for d in dims_content.split(",") if d.strip()]
        if not dims:
            return match.group(0)

        if max_batch_size > 0:
            if dims[0] == "-1":
                dims = dims[1:]
        else:
            if dims[0] != "-1" or len(dims) == 2:
                dims = ["-1"] + dims

        new_dims_str = ", ".join(dims)
        return f"dims: [{new_dims_str}]"

    return re.sub(r'dims\s*:\s*\[([^\]]+)\]', _fix_dims, block_str)


def serialize_config(cfg: dict) -> str:
    """Serialise a config dict back to protobuf text format."""
    lines: list[str] = []

    if "name" in cfg:
        lines.append(f'name: "{cfg["name"]}"')
    if "backend" in cfg:
        lines.append(f'backend: "{cfg["backend"]}"')
    if "max_batch_size" in cfg:
        lines.append(f'max_batch_size: {cfg["max_batch_size"]}')

    lines.append("")  # blank separator

    max_batch = int(cfg.get("max_batch_size", 0))

    # Preserved input / output blocks
    for block in cfg.get("_input_raw", []):
        lines.append(_adjust_dims_for_batch_size(block.strip(), max_batch))
    for block in cfg.get("_output_raw", []):
        lines.append(_adjust_dims_for_batch_size(block.strip(), max_batch))

    lines.append("")

    if "dynamic_batching" in cfg:
        db = cfg["dynamic_batching"]
        lines.append("dynamic_batching {")
        for s in db.get("preferred_batch_size", []):
            lines.append(f"  preferred_batch_size: {s}")
        if "max_queue_delay_microseconds" in db:
            lines.append(f'  max_queue_delay_microseconds: {db["max_queue_delay_microseconds"]}')
        lines.append("}")
        lines.append("")

    if "instance_group" in cfg:
        lines.append("instance_group [")
        for g in cfg["instance_group"]:
            lines.append("  {")
            if "count" in g:
                lines.append(f'    count: {g["count"]}')
            if "kind" in g:
                lines.append(f'    kind: {g["kind"]}')
            if "gpus" in g:
                gpu_str = ", ".join(str(x) for x in g["gpus"])
                lines.append(f'    gpus: [ {gpu_str} ]')
            lines.append("  }")
        lines.append("]")
        lines.append("")

    if "version_policy" in cfg:
        vp = cfg["version_policy"]
        if "latest" in vp:
            nv = vp["latest"].get("num_versions", 1)
            lines.append(f"version_policy {{ latest {{ num_versions: {nv} }} }}")

    return "\n".join(lines) + "\n"


# ──────────────────────────── merge ──────────────────────────────

def merge_config(existing: dict, updates: dict, allow_protected: bool = False) -> dict:
    """
    Merge *updates* into *existing*, silently ignoring protected fields (unless allow_protected=True).
    Private internal keys (starting with '_') are also preserved from existing.
    """
    result = dict(existing)
    for key, val in updates.items():
        if not allow_protected and (key in PROTECTED_FIELDS or key.startswith("_")):
            logger.debug(f"merge_config: skipping protected field '{key}'")
            continue
        result[key] = val
    return result


# ─────────────────────────── file I/O ────────────────────────────

def _config_path(model_repo: str, model_name: str) -> str:
    return os.path.join(model_repo, model_name, "config.pbtxt")


def read_model_config(model_repo: str, model_name: str) -> dict:
    path = _config_path(model_repo, model_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"config.pbtxt not found for model '{model_name}'")
    with open(path) as f:
        return parse_config(f.read())


def write_model_config(model_repo: str, model_name: str, cfg: dict) -> None:
    path = _config_path(model_repo, model_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(serialize_config(cfg))


def update_model_config(model_repo: str, model_name: str, updates: dict, allow_protected: bool = False) -> dict:
    """Read → merge → write.  Returns the merged dict."""
    existing = read_model_config(model_repo, model_name)
    merged = merge_config(existing, updates, allow_protected=allow_protected)
    write_model_config(model_repo, model_name, merged)
    return merged


# ─────────────────────── instance-group helpers ──────────────────

def get_instance_groups(model_repo: str, model_name: str) -> list[dict]:
    return read_model_config(model_repo, model_name).get("instance_group", [])


def update_instance_groups(model_repo: str, model_name: str, groups: list[dict]) -> dict:
    """Shortcut: replace only the instance_group block."""
    from gpu_manager import validate_instance_groups

    validate_instance_groups(groups)
    return update_model_config(model_repo, model_name, {"instance_group": groups})


# ─────────────────── base config generation ──────────────────────

def generate_base_config(
    model_name: str,
    model_type: str,
    task: str,
    num_classes: Optional[int] = None,
    overrides: Optional[dict] = None,
    input_dims: Optional[list[int]] = None,
    output0_dims: Optional[list[int]] = None,
    output1_dims: Optional[list[int]] = None,
) -> str:
    """
    Generate a valid config.pbtxt for a freshly uploaded model.

    Defaults: max_batch_size=8, dynamic_batching, 2x GPU instances.
    Overrides (if provided) are merged after generation.
    """
    is_yoloe_dynamic = model_type == "yoloe-dynamic"
    is_seg = task == "seg"
    channels = (num_classes + 4) if num_classes is not None else -1
    image_dims = input_dims or [-1, 3, -1, -1]
    pred_dims = output0_dims or [-1, channels, -1]
    proto_dims = output1_dims or [-1, 32, -1, -1]

    def dims_text(dims: list[int]) -> str:
        return ", ".join(str(int(d)) for d in dims)

    # max_batch_size: 0 — batch in tensor dims (matches existing model_repo style)
    # Separate input/output { } blocks so parse_config _extract_blocks preserves them.
    if is_yoloe_dynamic:
        input_block = f"""\
input {{
  name:      "images"
  data_type: TYPE_FP32
  dims: [{dims_text(image_dims)}]
}}
input {{
  name:      "prompt_embedding"
  data_type: TYPE_FP32
  dims: [-1, -1, 512]
}}"""
    else:
        input_block = f"""\
input {{
  name:      "images"
  data_type: TYPE_FP32
  dims: [{dims_text(image_dims)}]
}}"""

    if is_seg:
        output_block = f"""\
output {{
  name:      "output0"
  data_type: TYPE_FP32
  dims: [{dims_text(pred_dims)}]
}}
output {{
  name:      "output1"
  data_type: TYPE_FP32
  dims: [{dims_text(proto_dims)}]
}}"""
    else:
        output_block = f"""\
output {{
  name:      "output0"
  data_type: TYPE_FP32
  dims: [{dims_text(pred_dims)}]
}}"""

    from gpu_manager import default_gpu_index

    default_gpu = default_gpu_index()

    text = f"""\
name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: 0

{input_block}

{output_block}

instance_group [{{
  kind:  KIND_GPU
  gpus:  [{default_gpu}]
  count: 1
}}]
"""

    if overrides:
        from gpu_manager import validate_instance_groups

        if "instance_group" in overrides:
            validate_instance_groups(overrides["instance_group"])
        parsed = parse_config(text)
        merged = merge_config(parsed, overrides)
        return serialize_config(merged)
    return text
