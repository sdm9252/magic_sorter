"""Multi-object tracking using ByteTrack."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

try:
    from ultralytics.yolo.utils.trackers.byte_tracker import BYTETracker, STrack
except Exception as exc:  # pragma: no cover - fallback for environments without ultralytics
    BYTETracker = None  # type: ignore[assignment]
    STrack = None  # type: ignore[assignment]
    _import_error = exc
else:
    _import_error = None

from .detector import Detection


@dataclass
class Track:
    """Represents a tracked card."""
    track_id: int
    detection: Detection


class CardTracker:
    """Simple ByteTrack wrapper."""

    def __init__(self) -> None:
        if BYTETracker is None:
            raise RuntimeError(
                "ultralytics is required for CardTracker"  # pragma: no cover
            ) from _import_error
        self.tracker = BYTETracker()

    def update(self, detections: List[Detection], frame_shape: tuple[int, int]) -> List[Track]:
        """Update tracker with current frame detections."""
        if not detections:
            self.tracker.update(np.empty((0, 5)), frame_shape, frame_shape)
            return []
        dets = np.array([[d.x1, d.y1, d.x2, d.y2, d.confidence] for d in detections], dtype=float)
        online_targets = self.tracker.update(dets, frame_shape, frame_shape)
        tracks: List[Track] = []
        for t in online_targets:
            x1, y1, x2, y2 = t.tlbr
            det = Detection(x1, y1, x2, y2, t.score)
            tracks.append(Track(t.track_id, det))
        return tracks
