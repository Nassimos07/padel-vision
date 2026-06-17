"""Perspective-projected AR ground rings (exactly the notebook's ring).

A constant-radius circle is laid on the ground plane and projected through
tilt / yaw / roll + perspective, so it reads as painted on the court. Also
provides ``adjust_ring`` — an interactive OpenCV-trackbar tuner (the CLI
equivalent of the notebook's ipywidgets "Dial it in" cell).
"""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from . import calibration
from .config import Config
from .detection.detector import build_detector
from .video.io import grab_frame

PALETTE = sv.ColorPalette.from_hex(["#00E5FF", "#FF3DAE", "#FFD23F", "#7CFF6B"])
DEFAULT_RING: dict = {
    "radius": 70, "tilt": 68.0, "rot_y": 0.0, "rot_z": 0.0,
    "persp": 420.0, "depth": 0.45, "thickness": 2, "glow": 0.55,
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _rot(axis: str, deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def ring_points(center, p: dict, frame_h: int, n: int = 160) -> np.ndarray:
    """Project a constant-radius ground circle through tilt/yaw/roll + perspective."""
    cx, cy = center
    scale = 1.0 + p["depth"] * ((cy / frame_h) - 0.5) * 2.0
    r = p["radius"] * max(0.25, scale)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], 1) * r
    rot = _rot("x", p["tilt"]) @ _rot("y", p["rot_y"]) @ _rot("z", p["rot_z"])
    pts = pts @ rot.T
    f = p["persp"]
    factor = f / np.clip(f - pts[:, 2], 1e-3, None)
    return np.stack([cx + pts[:, 0] * factor, cy + pts[:, 1] * factor], 1)


def draw_ring(img, pts, color, thickness: int = 2, glow: float = 0.55):
    """Thin, broadcast-clean double ring with a soft glow."""
    pi = pts.astype(np.int32)
    if glow > 0:
        layer = np.zeros_like(img)
        cv2.polylines(layer, [pi], True, color, thickness + 7, cv2.LINE_AA)
        cv2.polylines(layer, [pi], True, color, 7, cv2.LINE_AA)     # dark base for contrast
        cv2.polylines(layer, [pi], True, color, 2, cv2.LINE_AA)            # colour rim

        img = cv2.addWeighted(img, 1.0, cv2.GaussianBlur(layer, (0, 0), 7), glow, 0)
    inner = ((pts - pts.mean(0)) * 0.80 + pts.mean(0)).astype(np.int32)
    cv2.polylines(img, [inner], True, (255, 255, 255), 1, cv2.LINE_AA)       # white inner highlight
    return img



def ground_rings(img, dets: sv.Detections, params: dict, frame_h: int):
    """Draw a projected ring under each detection, coloured by track id (or index)."""
    if dets.xyxy is None or len(dets) == 0:
        return img
    for i, (x1, _y1, x2, y2) in enumerate(dets.xyxy):
        idx = int(dets.tracker_id[i]) if dets.tracker_id is not None else i
        color = PALETTE.by_idx(idx).as_bgr()
        pts = ring_points(((x1 + x2) / 2, float(y2)), params, frame_h)
        img = draw_ring(img, pts, color, params["thickness"], params["glow"])
    return img


def _params_from_trackbars(win: str) -> dict:
    g = lambda name: cv2.getTrackbarPos(name, win)  # noqa: E731
    return {
        "radius": max(20, g("radius")),
        "tilt": float(g("tilt")),
        "rot_y": float(g("rotY+45") - 45),
        "rot_z": float(g("rotZ+180") - 180),
        "persp": float(max(120, g("persp"))),
        "depth": g("depth%") / 100.0,
        "thickness": max(1, g("thick")),
        "glow": g("glow%") / 100.0,
    }


def adjust_ring(video, frame: int = 300, model: str = "medium") -> None:
    """Interactively tune the AR ground-ring look on a frame and save it to calibration.

    Drag the sliders; press ``s`` to save the params (to data/calibration/<clip>.json,
    ``ring`` key) or ``q``/Esc to cancel. Needs a desktop display (WSLg / X server).
    """
    img = grab_frame(video, int(frame))
    h, w = img.shape[:2]

    cfg = Config().detector
    cfg.model = model
    dets = build_detector(cfg).detect(img)
    roi = calibration.roi(video)
    if roi is not None:
        dets = dets[sv.PolygonZone(polygon=roi).trigger(dets)]
    if len(dets) == 0:  # fallback: one sample ring near the bottom-centre
        dets = sv.Detections(
            xyxy=np.array([[w / 2 - 40, h * 0.55, w / 2 + 40, h * 0.62]], float),
            class_id=np.zeros(1, dtype=int),
        )

    p = dict(DEFAULT_RING)
    p.update(calibration.ring(video) or {})

    win = "padel-vision ring"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    cv2.imshow(win, img)
    cv2.waitKey(1)  # realize the window before adding trackbars (WSLg/Qt)

    def _noop(_):
        pass

    cv2.createTrackbar("radius", win, int(p["radius"]), 160, _noop)
    cv2.createTrackbar("tilt", win, int(p["tilt"]), 89, _noop)
    cv2.createTrackbar("rotY+45", win, int(p["rot_y"]) + 45, 90, _noop)
    cv2.createTrackbar("rotZ+180", win, int(p["rot_z"]) + 180, 360, _noop)
    cv2.createTrackbar("persp", win, int(p["persp"]), 2000, _noop)
    cv2.createTrackbar("depth%", win, int(p["depth"] * 100), 120, _noop)
    cv2.createTrackbar("thick", win, int(p["thickness"]), 6, _noop)
    cv2.createTrackbar("glow%", win, int(p["glow"] * 100), 100, _noop)

    while True:
        params = _params_from_trackbars(win)
        out = ground_rings(img.copy(), dets, params, h)
        cv2.putText(out, "drag sliders | 's' save | 'q' quit", (20, 40),
                    _FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, out)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow(win)
            print("cancelled — nothing saved")
            return
        if key == ord("s"):
            path = calibration.save_ring(video, params)
            cv2.destroyWindow(win)
            print(f"saved ring params -> {path}")
            return
