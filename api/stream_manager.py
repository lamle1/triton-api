"""
stream_manager.py — Per-stream frame-rate limiter.

Excess frames are dropped (not queued) to avoid back-pressure build-up.
Each WebSocket stream gets its own last-frame timestamp keyed by stream_id.

Usage:
    limiter = StreamRateLimiter(max_fps=10.0)
    if limiter.should_process(stream_id):
        # process frame
    else:
        # drop frame
"""
from __future__ import annotations

import time
import logging

logger = logging.getLogger(__name__)


class StreamRateLimiter:
    """
    Token-bucket-style FPS cap, one bucket per stream.

    Args:
        max_fps : maximum frames per second to process per stream (0 = unlimited)
    """

    def __init__(self, max_fps: float = 10.0) -> None:
        self.max_fps = max_fps
        self._min_interval: float = (1.0 / max_fps) if max_fps > 0 else 0.0
        self._last: dict[str, float] = {}  # stream_id → last processed timestamp

    def should_process(self, stream_id: str) -> bool:
        """
        Return True if the current frame should be processed,
        False if it should be dropped to stay within max_fps.
        """
        if self._min_interval == 0.0:
            return True

        now = time.monotonic()
        last = self._last.get(stream_id, 0.0)

        if now - last >= self._min_interval:
            self._last[stream_id] = now
            return True
        return False

    def remove_stream(self, stream_id: str) -> None:
        """Clean up state for a closed stream."""
        self._last.pop(stream_id, None)

    def active_streams(self) -> int:
        """Return number of streams with recorded activity."""
        return len(self._last)
