"""
encoder.py — YOLOETextEncoder

Loads a YOLOE .pt file once at startup and exposes encode() which calls
model.get_text_pe(texts) to produce prompt embeddings.

Compatible with:
  yoloe-v8s/m/l-seg    ✅
  yoloe-11s/m/l-seg    ✅
  yoloe-26s/m/l-seg    ✅

Embedding shape: [1, N, 512]  — N = number of class-name strings

Thread/async safety:
  encode() is synchronous and CPU-bound.  The caller (main.py) should
  run it in an executor thread so it doesn't block the event loop.
  The in-process dict cache is safe for concurrent reads; concurrent
  writes on the same key are idempotent.
"""
from __future__ import annotations

import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)


class YOLOETextEncoder:
    """
    Wraps a YOLOE .pt model to encode lists of class names into
    prompt embeddings suitable for Triton inference.

    Args:
        weights_path : path to yoloe-v8s-seg.pt (or v11 / v26 variant)
    """

    def __init__(self, weights_path: str) -> None:
        from ultralytics import YOLOE  # lazy import keeps startup fast if unused
        from yoloe_assets import ensure_mobileclip_asset

        ensure_mobileclip_asset()
        logger.info(f"Loading YOLOE encoder weights from {weights_path}")
        self._model = YOLOE(weights_path)
        self._model.eval()
        # Embedding cache: tuple[str, ...] → np.ndarray [1, N, 512]
        self._cache: dict[tuple[str, ...], np.ndarray] = {}
        logger.info("YOLOE encoder ready.")

    # ── public API ────────────────────────────────────────────────

    def encode(self, class_names: tuple[str, ...]) -> np.ndarray:
        """
        Encode class names to a prompt-embedding tensor.

        Args:
            class_names : tuple of strings  (tuple required for dict key)

        Returns:
            numpy float32 array  shape [1, N, 512]
        """
        if class_names in self._cache:
            return self._cache[class_names]

        with torch.no_grad():
            # get_text_pe is a method on the YOLOE wrapper object
            # Returns a torch.Tensor [1, N, 512]
            pe: torch.Tensor = self._model.get_text_pe(list(class_names))

        embedding = pe.cpu().numpy().astype(np.float32)
        self._cache[class_names] = embedding
        logger.debug(f"Encoded {len(class_names)} classes → shape {embedding.shape}")
        return embedding

    def clear_cache(self) -> None:
        """Evict all cached embeddings (call if memory is a concern)."""
        self._cache.clear()
