"""Detection utilities using Ultralytics YOLO models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - fallback for environments without ultralytics
    YOLO = None  # type: ignore[assignment]
    _import_error = exc
else:
    _import_error = None


@dataclass
class Detection:
    """Simple representation of a detection bounding box."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def as_xywh(self) -> tuple[float, float, float, float]:
        """Return bounding box in (x, y, w, h) format."""
        return self.x1, self.y1, self.width, self.height


class CardDetector:
    """Wrapper around a YOLO model for card detection."""

    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        if YOLO is None:
            raise RuntimeError(
                "ultralytics is required for CardDetector"  # pragma: no cover - runtime check
            ) from _import_error
        self.model = YOLO(model_path)

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Run detection on an image and return bounding boxes."""
        results = self.model(image, verbose=False)[0]
        detections: List[Detection] = []
        for x1, y1, x2, y2, conf, cls in results.boxes.data.tolist():
            detections.append(Detection(x1, y1, x2, y2, conf))
        return detections
