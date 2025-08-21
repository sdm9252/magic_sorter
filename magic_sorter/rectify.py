"""Utilities for rectifying detected card patches."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .detector import Detection


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Return points ordered as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def rectify_card(
    image: np.ndarray,
    detection: Detection,
    output_size: Tuple[int, int] = (224, 310),
) -> np.ndarray:
    """Warp the detection region to a top-down view.

    If a four-point contour can't be found, the function simply crops the
    bounding box region resized to ``output_size``.
    """
    x1, y1, x2, y2 = map(int, (detection.x1, detection.y1, detection.x2, detection.y2))
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            rect = _order_points(pts)
            w, h = output_size
            dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            return cv2.warpPerspective(roi, M, output_size)
    return cv2.resize(roi, output_size)
