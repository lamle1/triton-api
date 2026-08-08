"""
triton_client.py — Thin wrapper around tritonclient.grpc.

Handles two calling modes:
  • YOLOE dynamic : images  +  prompt_embedding → 2 inputs
  • Standard YOLO / YOLOE static : images only → 1 input

Input tensor names match what ultralytics ONNX export produces:
  "images"            FP32  [1, 3, 640, 640]
  "prompt_embedding"  FP32  [1, N, 512]       (YOLOE dynamic only)

Output names are collected dynamically from the Triton response so
they work regardless of the model's exact output naming.
"""
from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

logger = logging.getLogger(__name__)

# Triton input tensor names as exported by ultralytics
_IMG_INPUT_NAME = "images"
_PE_INPUT_NAME = "prompt_embedding"


class TritonGRPCClient:
    """
    Synchronous gRPC client for Triton Inference Server.

    Instantiate once at startup and reuse across requests.
    The underlying gRPC channel is managed by tritonclient.
    """

    def __init__(self, url: str) -> None:
        """
        Args:
            url : host:port of Triton gRPC endpoint, e.g. "triton:8001"
        """
        self._url = url
        self._client = grpcclient.InferenceServerClient(url=url, verbose=False)
        logger.info(f"TritonGRPCClient initialised → {url}")

    # ── inference ─────────────────────────────────────────────────

    def infer(
        self,
        model_name: str,
        image_tensor: np.ndarray,
        prompt_embedding: Optional[np.ndarray] = None,
        model_version: str = "1",
        return_named: bool = False,
    ) -> Union[list[np.ndarray], tuple[list[np.ndarray], dict[str, np.ndarray]]]:
        """
        Run inference on Triton.

        Args:
            model_name       : name of the model in Triton repository
            image_tensor     : float32  [1, 3, 640, 640]
            prompt_embedding : float32  [1, N, 512]  — None for single-input models
            model_version    : Triton model version string (default "1")

        Returns:
            List of numpy arrays, one per model output, in Triton output order.
        """
        inputs: list[grpcclient.InferInput] = []

        # ── input 0: images ───────────────────────────────────────
        inp_img = grpcclient.InferInput(
            _IMG_INPUT_NAME, list(image_tensor.shape), "FP32"
        )
        inp_img.set_data_from_numpy(image_tensor)
        inputs.append(inp_img)

        # ── input 1: prompt_embedding (YOLOE dynamic only) ────────
        if prompt_embedding is not None:
            pe = prompt_embedding.astype(np.float32)
            inp_pe = grpcclient.InferInput(
                _PE_INPUT_NAME, list(pe.shape), "FP32"
            )
            inp_pe.set_data_from_numpy(pe)
            inputs.append(inp_pe)

        try:
            response = self._client.infer(
                model_name=model_name,
                inputs=inputs,
                model_version=model_version,
            )
        except InferenceServerException as e:
            logger.error(f"Triton inference error [{model_name}]: {e}")
            raise

        meta = response.get_response()
        ordered: list[np.ndarray] = [
            response.as_numpy(out.name) for out in meta.outputs
        ]
        named: dict[str, np.ndarray] = {
            out.name: response.as_numpy(out.name) for out in meta.outputs
        }
        if return_named:
            return ordered, named
        return ordered

    # ── health / meta ─────────────────────────────────────────────

    def is_server_ready(self) -> bool:
        try:
            return bool(self._client.is_server_ready())
        except Exception:
            return False

    def is_model_ready(self, model_name: str, version: str = "1") -> bool:
        try:
            return bool(self._client.is_model_ready(model_name, version))
        except Exception:
            return False

    def get_model_metadata(self, model_name: str) -> dict:
        """Return raw Triton model metadata (inputs, outputs, versions)."""
        return self._client.get_model_metadata(model_name)
