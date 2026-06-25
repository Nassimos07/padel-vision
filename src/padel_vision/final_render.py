"""Notebook-style final cut renderer.

This ports the Padel Vision V2 notebook's direct OpenCV final-film path:
cached detections + mattes -> AR rings + player labels + stat cards +
bird's-eye heatmap panel -> H.264 video.
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict, deque
from functools import lru_cache
from math import hypot
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from . import calibration
from .rings import DEFAULT_RING, PALETTE, ring_points
from .track import TrackStabilizer
from .video.io import to_h264, video_info

UNIT = np.float32([(0, 0), (1, 0), (1, 1), (0, 1)])
NX, NY = 12, 8

POWER, GRID_ALPHA = 0.8, 0.9
HEATMAP_AT, HIDE_RINGS_AT, FADE = 3, 10000, 1.0
GAP, TRAIL = -380, 60
PPM_X, PPM_Y = 35.60, 35.75
SPEED_SMOOTH, SPEED_CAP_MS, MAX_GAP_S = 0.8, 9.0, 0.4
AVG_REF = 1.1
COVER_REF = 0.35 * NX * NY
CARD_W, CARD_H, CARD_MARGIN, CARD_VGAP, CARD_TOP = 300, 128, 30, 280, 120

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_DIR = Path("assets/fonts")
FONT_BOLD = FONT_DIR / "JetBrainsMono-Bold.ttf"
FONT_REG = FONT_DIR / "JetBrainsMono-Regular.ttf"


def _smooth(x: float) -> float:
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


def _player_bgr(tid: int) -> np.ndarray:
    try:
        return np.array(PALETTE.by_idx(int(tid)).as_bgr(), np.float32)
    except Exception:
        fallback = [(0, 229, 255), (174, 61, 255), (63, 210, 255), (107, 255, 124)]
        return np.array(fallback[int(tid) % len(fallback)], np.float32)


def _player_color(tid: int):
    return tuple(int(c) for c in _player_bgr(tid))


def ground_rings_fast(img: np.ndarray, dets: sv.Detections, params=None, frame_h=None):
    """Same notebook look, but blur the ring glow once for all players."""
    params = params or DEFAULT_RING
    frame_h = frame_h or img.shape[0]
    if dets.xyxy is None or len(dets) == 0:
        return img
    glow, thickness = params["glow"], params["thickness"]
    layer = np.zeros_like(img)
    inners = []
    for _i, (x1, _y1, x2, y2) in enumerate(dets.xyxy):
        pts = ring_points(((x1 + x2) / 2, float(y2)), params, frame_h)
        pi = pts.astype(np.int32)
        cv2.polylines(layer, [pi], True, (0, 0, 0), thickness + 7, cv2.LINE_AA)
        cv2.polylines(layer, [pi], True, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.polylines(layer, [pi], True, (0, 0, 0), 2, cv2.LINE_AA)
        inners.append(((pts - pts.mean(0)) * 0.80 + pts.mean(0)).astype(np.int32))
    if glow > 0:
        img = cv2.addWeighted(img, 1.0, cv2.GaussianBlur(layer, (0, 0), 7), glow, 0)
    for inner in inners:
        cv2.polylines(img, [inner], True, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def bring_to_front_fast(overlay: np.ndarray, frame: np.ndarray, matte: np.ndarray):
    m = matte[..., 0] if matte.ndim == 3 else matte
    ys, xs = np.where(m > 0.003)
    if len(ys) == 0:
        return overlay
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    mr = matte[y0:y1, x0:x1]
    out = overlay[y0:y1, x0:x1].astype(np.float32)
    src = frame[y0:y1, x0:x1].astype(np.float32)
    overlay[y0:y1, x0:x1] = (out * (1 - mr) + src * mr).astype(np.uint8)
    return overlay


class StableIdentityMapper:
    """Notebook-style stable visual IDs based on court-plane foot positions."""

    def __init__(self, homography_inv, max_age=45, max_cost=0.38, raw_bonus=0.10) -> None:
        self.hi = homography_inv
        self.max_age = int(max_age)
        self.max_cost = float(max_cost)
        self.raw_bonus = float(raw_bonus)
        self.next_id = 1
        self.state: dict[int, dict] = {}
        self.raw_to_stable: dict[int, int] = {}

    def _foot_uv(self, xyxy):
        x1, _y1, x2, y2 = xyxy
        pt = np.float32([[(x1 + x2) / 2, y2]]).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pt, self.hi).reshape(-1)

    def _predict(self, stable_id: int, frame_idx: int) -> np.ndarray:
        current = self.state[stable_id]
        dt = max(1, frame_idx - current["last_seen"])
        return current["uv"] + current["vel"] * min(dt, 8)

    def update(self, dets: sv.Detections, frame_idx: int) -> sv.Detections:
        if len(dets) == 0:
            for sid in list(self.state):
                if frame_idx - self.state[sid]["last_seen"] > self.max_age:
                    raw = self.state[sid].get("raw")
                    self.raw_to_stable.pop(raw, None)
                    del self.state[sid]
            return dets

        raw_ids = (
            dets.tracker_id.astype(int)
            if dets.tracker_id is not None
            else np.arange(len(dets), dtype=int)
        )
        uvs = np.array([self._foot_uv(xyxy) for xyxy in dets.xyxy], dtype=float)
        active = [
            sid
            for sid, state in self.state.items()
            if frame_idx - state["last_seen"] <= self.max_age
        ]
        assigned: dict[int, int] = {}
        used: set[int] = set()

        pairs = []
        for row, sid in enumerate(active):
            pred = self._predict(sid, frame_idx)
            for col, uv in enumerate(uvs):
                dist = float(np.linalg.norm(uv - pred))
                raw = int(raw_ids[col])
                if self.state[sid].get("raw") == raw:
                    dist -= self.raw_bonus
                elif self.raw_to_stable.get(raw) == sid:
                    dist -= self.raw_bonus * 0.5
                pairs.append((dist, row, col))
        for dist, row, col in sorted(pairs, key=lambda item: item[0]):
            sid = active[row]
            if dist > self.max_cost or col in assigned or sid in used:
                continue
            assigned[col] = sid
            used.add(sid)

        for col, raw in enumerate(raw_ids):
            if col in assigned:
                continue
            sid = self.raw_to_stable.get(int(raw))
            if sid in used:
                sid = None
            if sid is None:
                sid = self.next_id
                self.next_id += 1
            assigned[col] = sid
            used.add(sid)

        stable_ids = np.zeros(len(dets), dtype=int)
        for col, sid in assigned.items():
            uv = uvs[col]
            raw = int(raw_ids[col])
            old = self.state.get(sid)
            vel = (
                np.zeros(2, dtype=float)
                if old is None
                else 0.35 * ((uv - old["uv"]) / max(1, frame_idx - old["last_seen"]))
                + 0.65 * old["vel"]
            )
            self.state[sid] = {"uv": uv, "vel": vel, "raw": raw, "last_seen": frame_idx}
            self.raw_to_stable[raw] = sid
            stable_ids[col] = sid
        dets.tracker_id = stable_ids
        return dets


def _rfdetr_seg_detector(model_name: str = "nano", conf: float = 0.3):
    import rfdetr
    import torch

    table = {
        "nano": "RFDETRSegNano",
        "small": "RFDETRSegSmall",
        "medium": "RFDETRSegMedium",
        "large": "RFDETRSegLarge",
        "preview": "RFDETRSegPreview",
    }
    name = table.get(model_name.lower(), "RFDETRSegNano")
    if not hasattr(rfdetr, name):
        name = "RFDETRSegSmall"
    model = getattr(rfdetr, name)(device="cuda" if torch.cuda.is_available() else "cpu")

    def detect(frame_bgr: np.ndarray) -> sv.Detections:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        dets = model.predict(rgb, threshold=conf)
        dets = dets[dets.class_id == 1]
        dets.class_id = np.zeros(len(dets), dtype=int)
        return dets

    return detect


def _player_matte(dets: sv.Detections, shape) -> np.ndarray:
    h, w = shape[:2]
    matte = np.zeros((h, w), np.float32)
    if dets is not None and dets.mask is not None:
        for mask in dets.mask:
            matte[mask] = 1.0
    matte = cv2.erode(matte, np.ones((2, 2), np.uint8))
    matte = cv2.GaussianBlur(matte, (0, 0), 1.0)
    return matte[..., None]


def build_detection_cache(
    video: str | Path,
    output: str | Path | None = None,
    stride: int = 2,
    model: str = "nano",
    conf: float = 0.3,
) -> Path:
    """Build the notebook-format detection cache used by the final renderer."""
    video = Path(video)
    output = Path(output) if output else Path("data/processed/detect_cache.pkl")
    info = video_info(video)
    fps = int(round(info.fps))
    court = calibration.court(video)
    if court is None:
        raise SystemExit("no court calibration - run `padel-vision court adjust <video>` first")
    roi = calibration.roi(video)
    zone = sv.PolygonZone(polygon=roi if roi is not None else court.astype(np.int32))
    h_inv = cv2.getPerspectiveTransform(court, UNIT)

    detector = _rfdetr_seg_detector(model, conf)
    tracker = sv.ByteTrack(frame_rate=fps, lost_track_buffer=fps)
    identity = StableIdentityMapper(h_inv, max_age=45, max_cost=0.38, raw_bonus=0.10)
    stabilizer = TrackStabilizer()
    records = []

    cap = cv2.VideoCapture(str(video))
    total = int(info.total_frames)
    for frame_idx in tqdm(range(total), desc="Pass 1: detect + seg"):
        if frame_idx % max(1, int(stride)):
            cap.grab()
            continue
        ok, frame = cap.read()
        if not ok:
            break
        dets = detector(frame)
        dets = dets[zone.trigger(dets)]
        matte = _player_matte(dets, frame.shape)
        dets = tracker.update_with_detections(dets)
        if dets.tracker_id is not None and len(dets) > 4:
            counts = np.unique(dets.tracker_id, return_counts=True)
            keep_ids = counts[0][np.argsort(counts[1])[-4:]]
            dets = dets[np.isin(dets.tracker_id, keep_ids)]
        dets = identity.update(dets, frame_idx)
        dets = stabilizer.update(dets, frame_idx)
        _, png = cv2.imencode(".png", (matte[..., 0] * 255).astype(np.uint8))
        records.append(
            {
                "k": frame_idx,
                "boxes": (
                    dets.xyxy.astype(np.float32)
                    if len(dets)
                    else np.zeros((0, 4), np.float32)
                ),
                "ids": (
                    dets.tracker_id.astype(np.int32)
                    if dets.tracker_id is not None
                    else np.zeros(len(dets), np.int32)
                ),
                "matte": png.tobytes(),
            }
        )
    cap.release()

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as fh:
        pickle.dump(
            {
                "fps": fps,
                "stride": int(stride),
                "H": info.height,
                "W": info.width,
                "records": records,
            },
            fh,
        )
    print(
        f"cached {len(records)} frames -> {output} "
        f"({output.stat().st_size / 1e6:.1f} MB)"
    )
    return output


@lru_cache(maxsize=32)
def _font(path: str, size: int):
    path_obj = Path(path)
    if path_obj.exists():
        return ImageFont.truetype(str(path_obj), size)
    return ImageFont.load_default()


def _ink_for(bgr):
    b, g, r = bgr
    lum = 0.114 * b + 0.587 * g + 0.299 * r
    return (20, 20, 20) if lum > 140 else (245, 245, 245)


def _ink_rgb(rgb):
    r, g, b = rgb
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return (20, 20, 20) if lum > 140 else (245, 245, 245)


def _draw_player_labels(img, dets, frame_w: int, frame_h: int):
    if dets is None or dets.tracker_id is None or len(dets) == 0:
        return img
    font = _font(str(FONT_BOLD), 20)
    pad_x, pad_y, stem_h, gap = 12, 7, 9, 5
    chips = []
    for xyxy, tid in zip(dets.xyxy, dets.tracker_id, strict=False):
        x1, y1, x2, _y2 = xyxy
        text = f"P{int(tid)}"
        cx = int((x1 + x2) / 2)
        col_rgb = tuple(int(c) for c in _player_color(int(tid)))[::-1]
        left, top, right, bottom = font.getbbox(text)
        box_w = right - left + pad_x * 2
        box_h = bottom - top + pad_y * 2
        head_y = int(y1)
        x0 = int(np.clip(cx - box_w / 2, 2, frame_w - box_w - 2))
        y3 = int(np.clip(head_y - gap - stem_h, 2, frame_h - box_h - 2))
        chips.append((text, col_rgb, (x0, y3 - box_h, x0 + box_w, y3), cx, head_y))

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    for _text, _col, (x0, y0, x3, y3), _cx, _hy in chips:
        odraw.rounded_rectangle(
            (x0 + 3, y0 + 3, x3 + 3, y3 + 3),
            radius=(y3 - y0) // 2,
            fill=(0, 0, 0, 70),
        )
    pil = Image.alpha_composite(pil.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(pil)
    for text, col_rgb, (x0, y0, x3, y3), cx, head_y in chips:
        rad = (y3 - y0) // 2
        cxp = (x0 + x3) // 2
        draw.polygon(
            [(cxp - 6, y3 - 1), (cxp + 6, y3 - 1), (cxp, y3 + stem_h)],
            fill=col_rgb,
        )
        draw.rounded_rectangle((x0, y0, x3, y3), radius=rad, fill=col_rgb)
        draw.ellipse(
            (cx - 4, head_y - 4, cx + 4, head_y + 4),
            fill=col_rgb,
            outline=(255, 255, 255),
            width=1,
        )
        draw.text(
            (cxp, (y0 + y3) // 2),
            text,
            font=font,
            fill=_ink_rgb(col_rgb),
            anchor="mm",
        )
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def _draw_speed_tags(img, dets, speeds, frame_w: int):
    if dets.tracker_id is None:
        return img
    for xyxy, tid in zip(dets.xyxy, dets.tracker_id, strict=False):
        tid = int(tid)
        value = speeds.get(tid)
        if value is None:
            continue
        col = tuple(int(c) for c in _player_bgr(tid))
        cx, cy = int((xyxy[0] + xyxy[2]) / 2), int(xyxy[3]) + 8
        txt = f"{value:.1f} m/s"
        (tw, th), _ = cv2.getTextSize(txt, FONT, 0.6, 2)
        x0 = int(np.clip(cx - tw / 2 - 8, 0, frame_w - tw - 16))
        cv2.rectangle(
            img,
            (x0, cy),
            (x0 + tw + 16, cy + th + 12),
            (20, 20, 20),
            -1,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            img,
            (x0, cy),
            (x0 + tw + 16, cy + th + 12),
            col,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            txt,
            (x0 + 8, cy + th + 6),
            FONT,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return img


def _final_hud(img, _t, _phase):
    x, y, w, h = 24, 24, 360, 70
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (18, 18, 18), -1)
    return img


def _rrect(img, x, y, w, h, color, r=18, alpha=0.66):
    x2, y2 = x + w, y + h
    roi = img[y:y2, x:x2]
    panel = roi.copy()
    cv2.rectangle(panel, (r, 0), (w - r, h), color, -1)
    cv2.rectangle(panel, (0, r), (w, h - r), color, -1)
    for cx, cy in [(r, r), (w - r, r), (r, h - r), (w - r, h - r)]:
        cv2.circle(panel, (cx, cy), r, color, -1, cv2.LINE_AA)
    cv2.addWeighted(panel, alpha, roi, 1 - alpha, 0, roi)


def _stat_values(st):
    dist, tm = st["dist"], st["time"]
    avg = (dist / tm) if tm > 1e-3 else 0.0
    work = 100.0 * (
        0.55 * min(avg / AVG_REF, 1.0)
        + 0.45 * min(len(st["cells"]) / COVER_REF, 1.0)
    )
    return dist, avg, work


def _draw_stat_cards(img, stats, slot_of, slots_xy):
    text_ops = []
    sz_badge = int(CARD_H * 0.13)
    sz_header = int(CARD_H * 0.15)
    sz_value = int(CARD_H * 0.19)
    sz_label = int(CARD_H * 0.09)
    for tid, slot in slot_of.items():
        if slot is None or slot not in slots_xy or tid not in stats:
            continue
        x, y = slots_xy[slot]
        col = tuple(int(c) for c in _player_bgr(tid))
        dist, avg, work = _stat_values(stats[tid])
        _rrect(img, x, y, CARD_W, CARD_H, (255, 255, 255), 18, 0.9)
        cv2.circle(img, (x + 36, y + 38), 22, col, -1, cv2.LINE_AA)
        cv2.circle(img, (x + 36, y + 38), 22, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.line(
            img,
            (x + 70, y + 46),
            (x + CARD_W - 20, y + 46),
            col,
            2,
            cv2.LINE_AA,
        )
        bx0, bx1, by = x + 16, x + CARD_W - 16, y + CARD_H - 12
        cv2.line(img, (bx0, by), (bx1, by), (70, 70, 80), 4, cv2.LINE_AA)
        cv2.line(
            img,
            (bx0, by),
            (bx0 + int((bx1 - bx0) * work / 100), by),
            col,
            4,
            cv2.LINE_AA,
        )
        text_ops.append(
            (
                _font(str(FONT_BOLD), sz_badge),
                f"P{tid}",
                (x + 36, y + 38),
                _ink_for(col),
                "mm",
            )
        )
        text_ops.append(
            (
                _font(str(FONT_BOLD), sz_header),
                f"PLAYER {tid}",
                (x + 70, y + 30),
                (20, 20, 20),
                "lm",
            )
        )
        cols = [
            (f"{dist:.0f}", "Distance"),
            (f"{avg:.1f}", "Avg Speed"),
            (f"{work:.0f}", "Work rate"),
        ]
        col_w = (CARD_W - 24) // 3
        for ci, (value, label) in enumerate(cols):
            cxp = x + 12 + col_w * ci + col_w // 2
            suffix = " m" if label == "Distance" else " m/s" if label == "Avg Speed" else ""
            text_ops.append(
                (
                    _font(str(FONT_BOLD), sz_value),
                    value + suffix,
                    (cxp, y + 78),
                    (15, 15, 15),
                    "mm",
                )
            )
            text_ops.append(
                (
                    _font(str(FONT_BOLD), sz_label + 3),
                    label,
                    (cxp, y + 100),
                    (50, 50, 50),
                    "mm",
                )
            )
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for font, text, xy, fill, anchor in text_ops:
        draw.text(xy, text, font=font, fill=fill, anchor=anchor)
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def _dets_from(rec):
    boxes = rec["boxes"]
    if len(boxes) == 0:
        return sv.Detections.empty()
    return sv.Detections(xyxy=boxes, class_id=np.zeros(len(boxes), int), tracker_id=rec["ids"])


def render_final_cut(
    video: str | Path,
    cache: str | Path = "data/processed/detect_cache.pkl",
    bev_json: str | Path | None = None,
    court_map: str | Path = "data/court_map.jpg",
    output: str | Path = "data/processed/padel_final.mp4",
    show: bool = False,
) -> Path:
    """Render the attached notebook final layout from a notebook-format cache."""
    video = Path(video)
    cache = Path(cache)
    output = Path(output)
    bev_json = (
        Path(bev_json)
        if bev_json
        else Path("data/calibration") / f"{video.stem}_bev_points.json"
    )
    with cache.open("rb") as fh:
        cache_data = pickle.load(fh)
    records = cache_data["records"]
    fps = int(cache_data["fps"])
    stride = int(cache_data["stride"])
    frame_h, frame_w = int(cache_data["H"]), int(cache_data["W"])

    bev = json.loads(bev_json.read_text())
    src = np.float32([p["frame"] for p in bev["pairs"]])
    dst = np.float32([p["map"] for p in bev["pairs"]])
    h_bev, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    print(f"BEV homography from {len(src)} couples, inliers {int(mask.sum())}/{len(mask)}")

    court_map_img = cv2.imread(str(court_map))
    if court_map_img is None:
        raise FileNotFoundError(f"map not found: {court_map}")
    map_h, map_w = court_map_img.shape[:2]
    scale_map = frame_h / map_h
    panel_w = int(round(map_w * scale_map))
    map_panel0 = cv2.resize(court_map_img, (panel_w, frame_h))
    canvas_w = frame_w + GAP + panel_w
    final_w = canvas_w - 120
    slots_xy = {
        3: (CARD_MARGIN, CARD_TOP),
        1: (CARD_MARGIN, CARD_TOP + CARD_H + CARD_VGAP),
        2: (CARD_MARGIN, CARD_TOP + 200),
        0: (CARD_MARGIN, CARD_TOP + CARD_H + CARD_VGAP + 200),
    }

    def order_tlbr(points):
        by_y = sorted(range(4), key=lambda i: points[i][1])
        top = sorted(by_y[:2], key=lambda i: points[i][0])
        bottom = sorted(by_y[2:], key=lambda i: points[i][0])
        return [top[0], top[1], bottom[1], bottom[0]]

    def court_quad_indices(map_points: np.ndarray) -> list[int]:
        if len(map_points) < 4:
            raise ValueError("BEV calibration needs at least 4 point pairs")
        sums = map_points[:, 0] + map_points[:, 1]
        diffs = map_points[:, 0] - map_points[:, 1]
        return [
            int(np.argmin(sums)),   # top-left
            int(np.argmax(diffs)),  # top-right
            int(np.argmax(sums)),   # bottom-right
            int(np.argmin(diffs)),  # bottom-left
        ]

    grid_map_quad = bev.get("grid_map_quad")
    if grid_map_quad and len(grid_map_quad) == 4:
        map_quad = np.float32(grid_map_quad)
        print("Heatmap grid ROI loaded from grid_map_quad (TL, TR, BR, BL)")
    else:
        idx = court_quad_indices(dst)
        map_quad = dst[idx].astype(np.float32)
        order = order_tlbr(map_quad)
        map_quad = map_quad[order]
        print("Heatmap grid ROI inferred from BEV point extremes")
    h_map = cv2.getPerspectiveTransform(UNIT, map_quad)
    map_to_unit = cv2.getPerspectiveTransform(map_quad, UNIT)
    hi_bev = map_to_unit @ h_bev

    def cell_map(i, j):
        q = np.float32(
            [
                (i / NX, j / NY),
                ((i + 1) / NX, j / NY),
                ((i + 1) / NX, (j + 1) / NY),
                (i / NX, (j + 1) / NY),
            ]
        ).reshape(-1, 1, 2)
        return (cv2.perspectiveTransform(q, h_map).reshape(-1, 2) * scale_map).astype(
            np.int32
        )

    map_cells = [[cell_map(i, j) for i in range(NX)] for j in range(NY)]
    map_qmask = np.zeros((frame_h, panel_w), np.float32)
    cv2.fillConvexPoly(map_qmask, (map_quad * scale_map).astype(np.int32), 1.0)
    map_gridlines = np.zeros((frame_h, panel_w, 3), np.uint8)
    for row in map_cells:
        for cell in row:
            cv2.polylines(map_gridlines, [cell], True, (255, 255, 255), 1, cv2.LINE_AA)

    def bake_map_heat(grids):
        ids = sorted(grids.keys())
        if not ids:
            return np.zeros((frame_h, panel_w, 3), np.uint8), np.zeros(
                (frame_h, panel_w), np.float32
            )
        pstack = np.stack(
            [cv2.GaussianBlur(grids[tid].astype(np.float32), (0, 0), 0.8) for tid in ids],
            0,
        )
        colors = np.stack([_player_bgr(tid) for tid in ids], 0)
        cell_color = (pstack[..., None] * colors[:, None, None, :]).sum(0) / (
            pstack.sum(0)[..., None] + 1e-6
        )
        total = np.stack([grids[tid] for tid in ids], 0).sum(0)
        val = np.power(total / max(float(total.max()), 1e-6), POWER)
        val = cv2.GaussianBlur(val, (0, 0), 0.8)
        val = np.clip(val / max(float(val.max()), 1e-6), 0, 1)
        heat = np.zeros((frame_h, panel_w, 3), np.uint8)
        alpha = np.zeros((frame_h, panel_w), np.float32)
        for j in range(NY):
            for i in range(NX):
                v = float(val[j, i])
                if v < 0.035:
                    continue
                vv = v * v * (3 - 2 * v)
                cv2.fillConvexPoly(
                    heat,
                    map_cells[j][i],
                    tuple(int(c) for c in cell_color[j, i]),
                    cv2.LINE_AA,
                )
                cv2.fillConvexPoly(
                    alpha,
                    map_cells[j][i],
                    float(GRID_ALPHA * (0.12 + 0.78 * vv)),
                    cv2.LINE_AA,
                )
        return heat, alpha

    def compose_map_heat(bg, heat_pack, alpha):
        if alpha <= 0:
            return bg
        heat, heat_alpha = heat_pack
        area = (heat_alpha * alpha * map_qmask)[..., None]
        out = (bg.astype(np.float32) * (1 - area) + heat.astype(np.float32) * area).astype(
            np.uint8
        )
        return cv2.addWeighted(out, 1.0, map_gridlines, 0.18 * alpha, 0)

    def foot_map(x1, x2, y2):
        uv = cv2.perspectiveTransform(
            np.float32([[(x1 + x2) / 2, y2]]).reshape(-1, 1, 2), h_bev
        ).reshape(-1)
        if -25 <= uv[0] <= map_w + 25 and -25 <= uv[1] <= map_h + 25:
            return float(uv[0]), float(uv[1])
        return None

    def draw_map(positions, trails, speeds, heat_pack, heat_alpha):
        panel = compose_map_heat(map_panel0.copy(), heat_pack, heat_alpha)
        for tid, pos in positions.items():
            if pos is None:
                continue
            col = tuple(int(c) for c in _player_bgr(tid))
            x, y = pos[0] * scale_map, pos[1] * scale_map
            trails[tid].append((x, y))
            pts = list(trails[tid])
            for idx_pt in range(1, len(pts)):
                x0, y0 = pts[idx_pt - 1]
                x1, y1 = pts[idx_pt]
                cv2.line(
                    panel,
                    (int(x0), int(y0)),
                    (int(x1), int(y1)),
                    col,
                    max(1, int(4 * idx_pt / len(pts))),
                    cv2.LINE_AA,
                )
            cx, cy = int(x), int(y)
            cv2.circle(panel, (cx, cy), 11, (50, 50, 50), -1, cv2.LINE_AA)
            cv2.circle(panel, (cx, cy), 10, col, -1, cv2.LINE_AA)
            cv2.putText(
                panel,
                f"P{tid}",
                (cx + 14, cy - 10),
                FONT,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            _ = speeds
        cv2.putText(
            panel,
            "BIRD'S-EYE VIEW",
            (16, 36),
            FONT,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return panel

    ring_params = dict(DEFAULT_RING)
    ring_params.update(calibration.ring(video) or {})
    grids = defaultdict(lambda: np.zeros((NY, NX), np.float32))
    trails = defaultdict(lambda: deque(maxlen=TRAIL))
    last_pos, vel_ema = {}, {}
    stats = defaultdict(lambda: {"dist": 0.0, "time": 0.0, "cells": set()})
    slot_of = {}

    def slot(tid):
        if tid not in slot_of and len(slot_of) < 4:
            slot_of[tid] = len(slot_of)

    output.parent.mkdir(parents=True, exist_ok=True)
    out_info = sv.VideoInfo(width=final_w, height=frame_h, fps=max(1, fps // stride))
    cap = cv2.VideoCapture(str(video))
    rec_idx, want = 0, (records[0]["k"] if records else -1)
    stopped = False
    win = "Padel - final cut (q to stop)"
    if show:
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, 1600, int(1600 * frame_h / final_w))
        except cv2.error:
            show = False

    with sv.VideoSink(str(output), out_info) as sink:
        for frame_idx in tqdm(
            range(records[-1]["k"] + 1 if records else 0),
            desc="Pass 2: render + stats",
        ):
            if frame_idx != want:
                cap.grab()
                continue
            ok, frame = cap.read()
            if not ok:
                break
            rec = records[rec_idx]
            dets = _dets_from(rec)
            matte = (
                cv2.imdecode(
                    np.frombuffer(rec["matte"], np.uint8),
                    cv2.IMREAD_GRAYSCALE,
                ).astype(np.float32)[..., None]
                / 255.0
            )
            positions, speeds = {}, {}
            for xyxy, tid in zip(rec["boxes"], rec["ids"], strict=False):
                tid = int(tid)
                slot(tid)
                pos = foot_map(xyxy[0], xyxy[2], xyxy[3])
                positions[tid] = pos
                uvh = cv2.perspectiveTransform(
                    np.float32([[(xyxy[0] + xyxy[2]) / 2, xyxy[3]]]).reshape(-1, 1, 2),
                    hi_bev,
                ).reshape(-1)
                if 0 <= uvh[0] < 1 and 0 <= uvh[1] < 1:
                    ci, cj = int(uvh[0] * NX), int(uvh[1] * NY)
                    grids[tid][min(cj, NY - 1), min(ci, NX - 1)] += 1
                    stats[tid]["cells"].add((ci, cj))
                if pos is None:
                    continue
                xm, ym = pos[0] / PPM_X, pos[1] / PPM_Y
                if tid in last_pos:
                    xp, yp, prev_frame = last_pos[tid]
                    dt = (frame_idx - prev_frame) / fps
                    if 0 < dt <= MAX_GAP_S:
                        vx, vy = (xm - xp) / dt, (ym - yp) / dt
                        speed = hypot(vx, vy)
                        if speed > SPEED_CAP_MS:
                            vx, vy = vx * SPEED_CAP_MS / speed, vy * SPEED_CAP_MS / speed
                        ex, ey = vel_ema.get(tid, (vx, vy))
                        vel_ema[tid] = (
                            SPEED_SMOOTH * ex + (1 - SPEED_SMOOTH) * vx,
                            SPEED_SMOOTH * ey + (1 - SPEED_SMOOTH) * vy,
                        )
                        smoothed_speed = hypot(*vel_ema[tid])
                        stats[tid]["dist"] += smoothed_speed * dt
                        stats[tid]["time"] += dt
                last_pos[tid] = (xm, ym, frame_idx)
                ev = vel_ema.get(tid)
                speeds[tid] = hypot(*ev) if ev else 0.0

            t = frame_idx / fps
            heat_alpha = _smooth((t - HEATMAP_AT) / FADE)
            ring_alpha = 1.0 - _smooth((t - HIDE_RINGS_AT) / FADE)
            out = frame.copy()
            if ring_alpha > 0 and len(dets):
                rings = ground_rings_fast(out.copy(), dets, ring_params, frame_h)
                out = cv2.addWeighted(rings, ring_alpha, out, 1.0 - ring_alpha, 0)
            out = bring_to_front_fast(out, frame, matte)
            if len(dets):
                out = _draw_player_labels(out, dets, frame_w, frame_h)
                out = _draw_speed_tags(out, dets, speeds, frame_w)
            out = _final_hud(out, t, "LIVE TRACKING" if t < HEATMAP_AT else "LIVE HEATMAP")
            out = _draw_stat_cards(out, stats, slot_of, slots_xy)
            panel = draw_map(positions, trails, speeds, bake_map_heat(grids), heat_alpha)
            canvas = np.zeros((frame_h, canvas_w, 3), np.uint8)
            canvas[:, frame_w + GAP :] = panel
            canvas[:, : frame_w - 280] = out[:, : frame_w - 280]
            canvas = canvas[:, :-120]
            sink.write_frame(canvas)
            if show:
                cv2.imshow(win, canvas)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    stopped = True
                    break
            rec_idx += 1
            want = records[rec_idx]["k"] if rec_idx < len(records) else -1
    cap.release()
    if show:
        cv2.destroyAllWindows()
    playable = to_h264(output, output.with_name(f"{output.stem}_h264.mp4"))
    print("Final cut ->", "stopped early" if stopped else "done", "->", playable)
    return playable
