"""Live player tracking with the notebook's full AR overlay."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from supervision.annotators.utils import ColorLookup

from . import calibration
from .config import Config
from .detection.detector import build_detector
from .rings import DEFAULT_RING, PALETTE, ground_rings

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LK = ColorLookup.TRACK


class TrackStabilizer:
    """EMA-smooth tracked boxes and coast briefly through missing detections."""

    def __init__(self, smoothing: float = 0.35, hold_frames: int = 30) -> None:
        self.alpha = float(smoothing)
        self.hold = int(hold_frames)
        self._state: dict[int, dict] = {}

    def update(self, detections: sv.Detections, frame_idx: int) -> sv.Detections:
        if detections.tracker_id is not None:
            confs = (
                detections.confidence
                if detections.confidence is not None
                else np.ones(len(detections), dtype=float)
            )
            for xyxy, tracker_id, conf in zip(
                detections.xyxy, detections.tracker_id, confs, strict=False
            ):
                tracker_id = int(tracker_id)
                xyxy = np.asarray(xyxy, dtype=float)
                state = self._state.get(tracker_id)
                if state is None:
                    state = {"xyxy": xyxy.copy()}
                else:
                    state["xyxy"] = self.alpha * xyxy + (1 - self.alpha) * state["xyxy"]
                state["conf"] = float(conf)
                state["last_seen"] = int(frame_idx)
                self._state[tracker_id] = state

        ids, boxes, confs = [], [], []
        for tracker_id, state in list(self._state.items()):
            if frame_idx - state["last_seen"] > self.hold:
                del self._state[tracker_id]
                continue
            ids.append(tracker_id)
            boxes.append(state["xyxy"])
            confs.append(state["conf"])

        if not ids:
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=np.array(boxes, dtype=float),
            confidence=np.array(confs, dtype=float),
            class_id=np.zeros(len(ids), dtype=int),
            tracker_id=np.array(ids, dtype=int),
        )


class ForegroundMatte:
    """YOLO person segmentation matte used to paste players above AR graphics."""

    def __init__(self, polygon: np.ndarray | None = None, model: str = "yolo11n-seg.pt") -> None:
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.polygon = polygon

    def __call__(self, frame: np.ndarray, erode: int = 2, feather: float = 1.0) -> np.ndarray:
        result = self.model(frame, verbose=False, classes=[0], retina_masks=True)[0]
        mask = np.zeros(frame.shape[:2], np.float32)
        if result.masks is not None:
            for poly, box in zip(result.masks.xy, result.boxes.xyxy.cpu().numpy(), strict=False):
                x1, _y1, x2, y2 = box
                foot = (float((x1 + x2) / 2), float(y2))
                if self.polygon is None or cv2.pointPolygonTest(self.polygon, foot, False) >= 0:
                    cv2.fillPoly(mask, [poly.astype(np.int32)], 1.0)
        if erode:
            mask = cv2.erode(mask, np.ones((erode, erode), np.uint8))
        if feather:
            mask = cv2.GaussianBlur(mask, (0, 0), feather)
        return mask[..., None]


def bring_to_front(
    overlay: np.ndarray, frame: np.ndarray, matte: np.ndarray | None = None
) -> np.ndarray:
    """Re-paste original player pixels on top of an overlay."""
    if matte is None:
        return overlay
    return (overlay.astype(np.float32) * (1 - matte) + frame.astype(np.float32) * matte).astype(
        np.uint8
    )


def draw_court(img: np.ndarray, polygon: np.ndarray | None, alpha: float = 0.45) -> np.ndarray:
    """Draw the saved play-area outline if one exists."""
    if polygon is None:
        return img
    overlay = img.copy()
    cv2.polylines(overlay, [polygon.astype(np.int32)], True, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def hud(img: np.ndarray, n_players: int, t_sec: float, fps: float, model: str) -> np.ndarray:
    """Translucent broadcast-style info panel, matching the v1 notebook."""
    x, y, w, h = 24, 24, 380, 112
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (18, 18, 18), -1)
    img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        img, "PADEL ANALYTICS", (x + 18, y + 38), _FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        img, f"players tracked: {n_players}", (x + 18, y + 70),
        _FONT, 0.6, (255, 255, 0), 2, cv2.LINE_AA,
    )
    cv2.putText(
        img, f"{t_sec:5.2f}s | {fps:4.1f} fps | {model}",
        (x + 18, y + 96), _FONT, 0.55, (210, 210, 210), 1, cv2.LINE_AA,
    )
    return img


class FullOverlayRenderer:
    """Compose court outline, trail, AR rings, markers, foreground matte, and HUD."""

    def __init__(
        self,
        frame_h: int,
        polygon: np.ndarray | None,
        ring_params: dict,
        fps: int,
        trail: bool = False,
        labels: bool = False,
    ) -> None:
        self.frame_h = frame_h
        self.polygon = polygon
        self.ring_params = ring_params
        self.trail = bool(trail)
        self.labels = bool(labels)
        self.triangle = sv.TriangleAnnotator(color=PALETTE, base=22, height=18, color_lookup=_LK)
        self.trace = sv.TraceAnnotator(
            color=PALETTE,
            thickness=3,
            trace_length=max(1, int(fps)),
            position=sv.Position.BOTTOM_CENTER,
            color_lookup=_LK,
        )
        self.label = sv.LabelAnnotator(
            color=PALETTE,
            text_color=sv.Color.BLACK,
            text_scale=0.55,
            text_position=sv.Position.BOTTOM_CENTER,
            color_lookup=_LK,
        )

    def render(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        t_sec: float,
        live_fps: float,
        model: str,
        matte: np.ndarray | None = None,
        base: np.ndarray | None = None,
    ) -> np.ndarray:
        out = draw_court((base if base is not None else frame).copy(), self.polygon)
        if self.trail:
            out = self.trace.annotate(out, detections)
        out = ground_rings(out, detections, self.ring_params, self.frame_h)
        out = self.triangle.annotate(out, detections)
        if self.labels and detections.tracker_id is not None and len(detections):
            labels = [f"P{int(t)}" for t in detections.tracker_id]
            out = self.label.annotate(out, detections, labels)
        out = bring_to_front(out, frame, matte)
        return hud(out, len(detections), t_sec, live_fps, model)


def _filter_polygon(video: str | Path) -> np.ndarray | None:
    roi = calibration.roi(video)
    if roi is not None:
        return roi
    court = calibration.court(video)
    return court.astype(np.int32) if court is not None else None


def track_players(
    video: str | Path,
    frame: int = 0,
    conf: float = 0.5,
    model: str = "medium",
    stride: int = 1,
    smoothing: float = 0.35,
    hold_frames: int = 30,
    foreground: bool = True,
    foreground_model: str = "yolo11n-seg.pt",
    foreground_stride: int = 1,
    trail: bool = False,
    labels: bool = False,
) -> None:
    """Show the v1 notebook's full AR tracking overlay in a live OpenCV window."""
    polygon = _filter_polygon(video)
    if polygon is not None:
        zone = sv.PolygonZone(polygon=polygon)
        print(f"loaded tracking polygon ({len(polygon)} points) from {calibration.path_for(video)}")
    else:
        zone = None
        print("no ROI/court calibration found - tracking on the whole frame")

    ring_params = dict(DEFAULT_RING)
    ring_params.update(calibration.ring(video) or {})

    cfg = Config().detector
    cfg.confidence = conf
    cfg.model = model
    detector = build_detector(cfg)

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise FileNotFoundError(f"could not read frame {frame} from {video!r}")

    fps = int(round(cap.get(cv2.CAP_PROP_FPS) or 30))
    renderer = FullOverlayRenderer(
        first.shape[0], polygon, ring_params, fps, trail=trail, labels=labels
    )
    tracker = sv.ByteTrack(frame_rate=max(1, fps), lost_track_buffer=max(1, fps))
    stabilizer = TrackStabilizer(smoothing=smoothing, hold_frames=hold_frames)
    matte_builder = ForegroundMatte(polygon, foreground_model) if foreground else None

    win = "padel-vision track players  (q / Esc to quit)"
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
    except cv2.error as e:
        cap.release()
        raise RuntimeError(
            "track players needs a desktop display (WSLg / X server); none was found"
        ) from e

    stride = max(1, int(stride))
    foreground_stride = max(1, int(foreground_stride))
    live_fps, prev = 0.0, time.time()
    idx = int(frame)
    last = sv.Detections.empty()
    last_matte = None

    try:
        while True:
            current = first
            first = None

            if (idx - int(frame)) % stride == 0:
                dets = detector.detect(current)
                if zone is not None:
                    dets = dets[zone.trigger(dets)]
                dets = tracker.update_with_detections(dets)
                last = stabilizer.update(dets, idx)
            else:
                last = stabilizer.update(sv.Detections.empty(), idx)

            if matte_builder is not None and (idx - int(frame)) % foreground_stride == 0:
                last_matte = matte_builder(current)

            now = time.time()
            dt = now - prev
            prev = now
            if dt > 0:
                live_fps = 0.9 * live_fps + 0.1 * (1.0 / dt)
            t_sec = idx / max(1, fps)

            out = renderer.render(current, last, t_sec, live_fps, model, matte=last_matte)
            cv2.imshow(win, out)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break

            ok, next_frame = cap.read()
            if not ok:
                break
            first = next_frame
            idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
