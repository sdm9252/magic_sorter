"""Simple webcam demo for detecting and identifying trading cards."""
from __future__ import annotations

import cv2

from .detector import CardDetector
from .tracker import CardTracker
from .rectify import rectify_card
from .retrieval import CardGallery


def run(gallery_dir: str) -> None:
    detector = CardDetector()
    tracker = CardTracker()
    gallery = CardGallery.from_directory(gallery_dir)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame.shape[:2])
            for track in tracks:
                patch = rectify_card(frame, track.detection)
                candidates = gallery.phash_shortlist(patch)
                if candidates:
                    results = gallery.search(patch, top_k=1)
                    if results:
                        path, score = results[0]
                        x1, y1, x2, y2 = map(int, (
                            track.detection.x1,
                            track.detection.y1,
                            track.detection.x2,
                            track.detection.y2,
                        ))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"{path.stem} {score:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )
            cv2.imshow("magic_sorter", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Magic card identifier demo")
    parser.add_argument("gallery", help="Directory of gallery images")
    args = parser.parse_args()
    run(args.gallery)
