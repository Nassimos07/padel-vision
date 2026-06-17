"""Render a stabilized AR-overlay reel from a padel clip.

Standalone companion to notebooks/tutorials/padel_ar_showcase_v1.ipynb. Same pipeline
(detect -> court filter -> track -> AR ground rings), plus two extras that make
the overlay broadcast-smooth:

  1. EMA smoothing of each track's box, so the rings don't jitter while a player
     moves (tune with ``SMOOTHING``).
  2. "Coasting": when the detector/tracker drops a player for a few frames, we
     keep drawing it at its last known position instead of letting the ring
     blink out (tune with ``HOLD_FRAMES``).

Run from the repo root:
    python scripts/render_reel.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from supervision.annotators.utils import ColorLookup
from tqdm import tqdm

from padel_vision.config import Config
from padel_vision.court import load_corners
from padel_vision.detection.detector import build_detector
from padel_vision.video.io import to_h264

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Settings — tweak these
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data" / "raw" / "padel_clip.mp4"
OUT = ROOT / "data" / "processed"

START_SEC, DURATION_SEC = 4.0, 6.0          # which slice of the match to render

# Stabilization
SMOOTHING = 0.35     # EMA weight on the new measurement (lower = smoother, more lag)
HOLD_FRAMES = 30     # keep drawing a lost track this many frames using its last box

# Court region (image pixels) — detections whose feet fall outside are dropped.
# Reuse the saved corners from `padel-vision court adjust` if present, else a default.
_CORNERS = ROOT / "notebooks" / "tutorials" / "court_corners.txt"
COURT = (
    load_corners(_CORNERS).astype(np.int32)
    if _CORNERS.exists()
    else np.array([(640, 215), (1270, 215), (1480, 470),
                   (1520, 1060), (380, 1060), (430, 470)], dtype=np.int32)
)

# AR ground-ring look (matches the notebook's tuned defaults)
RING = dict(radius=70, tilt=68.0, rot_y=0.0, rot_z=0.0, persp=420.0, depth=0.45,
            thickness=2, glow=0.55)

PALETTE = sv.ColorPalette.from_hex(["#00E5FF", "#FF3DAE", "#FFD23F", "#7CFF6B"])
LK = ColorLookup.TRACK


# --------------------------------------------------------------------------- #
# Track stabilizer: EMA smoothing + coast-through-lost-frames
# --------------------------------------------------------------------------- #
class TrackStabilizer:
    """Smooths each track's box and keeps it alive briefly after it's lost.

    Feed it the per-frame tracked detections; it returns a new ``sv.Detections``
    where every box is EMA-smoothed and any track missing this frame is re-emitted
    at its last position for up to ``hold_frames`` frames.
    """

    def __init__(self, smoothing: float = SMOOTHING, hold_frames: int = HOLD_FRAMES):
        self.alpha = float(smoothing)
        self.hold = int(hold_frames)
        self._state: dict[int, dict] = {}   # tracker_id -> {xyxy, conf, last_seen}

    def update(self, detections: sv.Detections, frame_idx: int) -> sv.Detections:
        if detections.tracker_id is not None:
            confs = (detections.confidence
                     if detections.confidence is not None
                     else np.ones(len(detections)))
            for xyxy, tid, conf in zip(detections.xyxy, detections.tracker_id, confs, strict=False):
                tid = int(tid)
                xyxy = np.asarray(xyxy, dtype=float)
                st = self._state.get(tid)
                if st is None:                                   # first sighting
                    st = {"xyxy": xyxy.copy()}
                else:                                            # EMA toward new box
                    st["xyxy"] = self.alpha * xyxy + (1 - self.alpha) * st["xyxy"]
                st["conf"] = float(conf)
                st["last_seen"] = frame_idx
                self._state[tid] = st

        ids, boxes, confs = [], [], []
        for tid, st in list(self._state.items()):
            if frame_idx - st["last_seen"] > self.hold:          # expired -> forget
                del self._state[tid]
                continue
            ids.append(tid)
            boxes.append(st["xyxy"])
            confs.append(st["conf"])

        if not ids:
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=np.array(boxes, dtype=float),
            confidence=np.array(confs, dtype=float),
            class_id=np.zeros(len(ids), dtype=int),
            tracker_id=np.array(ids, dtype=int),
        )


# --------------------------------------------------------------------------- #
# AR ground ring (perspective-projected, ROI-drawn for speed)
# --------------------------------------------------------------------------- #
def _rot(axis: str, deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def ring_points(center, p, frame_h, n=160) -> np.ndarray:
    """Project a constant-radius ground circle through tilt/yaw/roll + perspective."""
    cx, cy = center
    scale = 1.0 + p["depth"] * ((cy / frame_h) - 0.5) * 2.0
    r = p["radius"] * max(0.25, scale)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], 1) * r
    R = _rot("x", p["tilt"]) @ _rot("y", p["rot_y"]) @ _rot("z", p["rot_z"])
    pts = pts @ R.T
    f = p["persp"]
    factor = f / np.clip(f - pts[:, 2], 1e-3, None)
    return np.stack([cx + pts[:, 0] * factor, cy + pts[:, 1] * factor], 1)


def draw_ring(img, pts, color, thickness=2, glow=0.55):
    """Thin, broadcast-clean double ring with a soft glow (drawn on a local ROI)."""
    pi = pts.astype(np.int32)
    h, w = img.shape[:2]
    m = thickness + 24
    x0, y0 = max(0, pi[:, 0].min() - m), max(0, pi[:, 1].min() - m)
    x1, y1 = min(w, pi[:, 0].max() + m), min(h, pi[:, 1].max() + m)
    if x1 <= x0 or y1 <= y0:
        return img
    roi = img[y0:y1, x0:x1]
    p = pi - [x0, y0]
    if glow > 0:
        layer = np.zeros_like(roi)
        cv2.polylines(layer, [p], True, color, thickness + 7, cv2.LINE_AA)
        roi[:] = cv2.addWeighted(roi, 1.0, cv2.GaussianBlur(layer, (0, 0), 7), glow, 0)
    cv2.polylines(roi, [p], True, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.polylines(roi, [p], True, color, thickness, cv2.LINE_AA)
    inner = ((p - p.mean(0)) * 0.80 + p.mean(0)).astype(np.int32)
    cv2.polylines(roi, [inner], True, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def ground_rings(img, dets, frame_h):
    if dets.tracker_id is None:
        return img
    for xyxy, tid in zip(dets.xyxy, dets.tracker_id, strict=False):
        x1, y1, x2, y2 = xyxy
        color = PALETTE.by_idx(int(tid)).as_bgr()
        pts = ring_points(((x1 + x2) / 2, float(y2)), RING, frame_h)
        img = draw_ring(img, pts, color, RING["thickness"], RING["glow"])
    return img


# --------------------------------------------------------------------------- #
# Supporting overlay (court outline, trail, pointer, label, HUD)
# --------------------------------------------------------------------------- #
def draw_court(img, alpha=0.45, thickness=2):
    ov = img.copy()
    cv2.polylines(ov, [COURT], True, (255, 255, 255), thickness, cv2.LINE_AA)
    return cv2.addWeighted(ov, alpha, img, 1 - alpha, 0)


def hud(img, n_players, t_sec):
    x, y, w, h = 24, 24, 380, 104
    ov = img.copy()
    cv2.rectangle(ov, (x, y), (x + w, y + h), (18, 18, 18), -1)
    img = cv2.addWeighted(ov, 0.55, img, 0.45, 0)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "PADEL ANALYTICS", (x + 18, y + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"players tracked: {n_players}", (x + 18, y + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, f"t = {t_sec:5.2f}s", (x + 18, y + 94),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def main() -> None:
    info = sv.VideoInfo.from_video_path(str(VIDEO))
    fps = int(round(info.fps))
    frame_h = info.height
    start = int(START_SEC * fps)
    n_frames = int(DURATION_SEC * fps)

    detector = build_detector(Config().detector)
    zone = sv.PolygonZone(polygon=COURT)
    tracker = sv.ByteTrack(frame_rate=fps, lost_track_buffer=fps)
    stabilizer = TrackStabilizer()

    triangle = sv.TriangleAnnotator(color=PALETTE, base=22, height=18, color_lookup=LK)
    trace = sv.TraceAnnotator(color=PALETTE, thickness=3, trace_length=fps,
                              position=sv.Position.BOTTOM_CENTER, color_lookup=LK)
    label = sv.LabelAnnotator(color=PALETTE, text_color=sv.Color.BLACK, text_scale=0.55,
                              text_position=sv.Position.BOTTOM_CENTER, color_lookup=LK)

    def render(frame, dets, t_sec):
        out = draw_court(frame.copy())
        out = trace.annotate(out, dets)
        out = ground_rings(out, dets, frame_h)
        out = triangle.annotate(out, dets)
        if dets.tracker_id is not None and len(dets):
            out = label.annotate(out, dets, [f"P{int(t)}" for t in dets.tracker_id])
        return hud(out, len(dets), t_sec)

    OUT.mkdir(parents=True, exist_ok=True)
    reel = OUT / "padel_ar_reel.mp4"
    out_info = sv.VideoInfo(width=info.width, height=info.height, fps=fps)

    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    with sv.VideoSink(str(reel), out_info) as sink:
        for i in tqdm(range(n_frames), desc="Rendering reel"):
            ok, f = cap.read()
            if not ok:
                break
            d = detector.detect(f)
            d = d[zone.trigger(d)]
            d = tracker.update_with_detections(d)
            d = stabilizer.update(d, i)            # <-- smooth + coast through lost frames
            sink.write_frame(render(f, d, t_sec=START_SEC + i / fps))
    cap.release()

    playable = to_h264(reel, OUT / "padel_ar_reel_h264.mp4")
    print(f"Reel: {playable}")


if __name__ == "__main__":
    main()
