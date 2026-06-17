"""Real-time player detection, filtered by the saved ROI.

Loads the clip's ROI from the calibration store (``padel-vision roi adjust``) and
keeps only detections whose feet fall inside it; with no ROI it detects on the
whole frame. Shows a live OpenCV window (needs a desktop display).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import supervision as sv

from . import calibration
from .config import Config
from .detection.detector import build_detector

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def detect_players(video: str | Path, frame: int = 0, conf: float = 0.5) -> None:
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
    detector = build_detector(cfg)
    box = sv.RoundBoxAnnotator(thickness=2)

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
    try:
        while True:
            ok, frame_img = cap.read()
            if not ok:
                break
            dets = detector.detect(frame_img)
            if zone is not None:
                dets = dets[zone.trigger(dets)]
            out = box.annotate(frame_img.copy(), dets)
            if roi is not None:
                cv2.polylines(out, [roi], True, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(out, f"players: {len(dets)}", (20, 44), _FONT, 1.0, (255, 255, 0), 2)
            cv2.imshow(win, out)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
