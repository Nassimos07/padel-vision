"""Real-time player detection, filtered by the saved ROI.

Loads the clip's ROI from the calibration store (``padel-vision roi adjust``) and
keeps only detections whose feet fall inside it; with no ROI it detects on the
whole frame. Shows a live OpenCV window (needs a desktop display).

Detection runs every ``stride`` frames (reusing the last result in between) so the
video plays smoothly even on a modest GPU; pick a lighter ``model`` for more speed.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import supervision as sv

from . import calibration
from .config import Config
from .detection.detector import build_detector

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def detect_players(
    video: str | Path, frame: int = 0, conf: float = 0.5,
    model: str = "medium", stride: int = 2,
) -> None:
    """Detect players in real time, filtering to the saved ROI if there is one."""
    roi = calibration.roi(video)
    if roi is not None:
        zone = sv.PolygonZone(polygon=roi)
        print(f"loaded ROI ({len(roi)} points) from {calibration.path_for(video)}")
    else:
        zone = None
        print("no ROI data found — detecting on the whole frame")

    cfg = Config().detector
    cfg.confidence = conf
    cfg.model = model
    detector = build_detector(cfg)
    box = sv.RoundBoxAnnotator(thickness=2)
    stride = max(1, int(stride))

    win = "padel-vision detect players  (q / Esc to quit)"
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
    except cv2.error as e:  # no display available
        raise RuntimeError(
            "detect players needs a desktop display (WSLg / X server); none was found"
        ) from e

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    last = sv.Detections.empty()
    i, prev, fps = 0, time.time(), 0.0
    try:
        while True:
            ok, frame_img = cap.read()
            if not ok:
                break
            if i % stride == 0:                       # detect every `stride`-th frame
                dets = detector.detect(frame_img)
                last = dets[zone.trigger(dets)] if zone is not None else dets
            i += 1

            out = box.annotate(frame_img.copy(), last)
            if roi is not None:
                cv2.polylines(out, [roi], True, (0, 255, 255), 2, cv2.LINE_AA)
            now = time.time()
            dt = now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            label = f"players: {len(last)} | {fps:4.1f} fps | {model} (GPU)"
            cv2.putText(out, label, (20, 44), _FONT, 0.9, (255, 255, 0), 2)
            cv2.imshow(win, out)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
