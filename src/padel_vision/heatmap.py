"""Zonal court heatmap.

Accumulate player foot positions over a clip, bin them into a perspective grid
defined by the saved court corners, and render a broadcast-style green→red
heatmap. Needs ``court adjust`` first; uses the ROI (if any) to drop the crowd.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

from . import calibration
from .config import Config
from .detection.detector import build_detector
from .video.io import grab_frame, video_info

_UNIT = np.float32([(0, 0), (1, 0), (1, 1), (0, 1)])


def _green_red_lut() -> np.ndarray:
    """256-entry BGR LUT: green (low) → yellow → red (high)."""
    n = 256
    lut = np.zeros((n, 3), np.uint8)
    h = n // 2
    lut[:h, 1] = 255
    lut[:h, 2] = np.linspace(0, 255, h)        # green → yellow
    lut[h:, 2] = 255
    lut[h:, 1] = np.linspace(255, 0, n - h)    # yellow → red
    return lut


def _cell(Hc, i, j, nx, ny) -> np.ndarray:
    u0, u1, v0, v1 = i / nx, (i + 1) / nx, j / ny, (j + 1) / ny
    corners = np.float32([(u0, v0), (u1, v0), (u1, v1), (u0, v1)]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(corners, Hc).reshape(-1, 2).astype(np.int32)


def _render(bg, grid, quad, Hc, nx, ny, power, alpha) -> np.ndarray:
    h, w = bg.shape[:2]
    cells = [[_cell(Hc, i, j, nx, ny) for i in range(nx)] for j in range(ny)]
    g = cv2.GaussianBlur(grid, (0, 0), 0.8)
    g = g / g.max() if g.max() > 0 else g
    g = np.power(g, power)
    lut = _green_red_lut()

    qmask = np.zeros((h, w), np.float32)
    cv2.fillConvexPoly(qmask, quad.astype(np.int32), 1.0)
    heat = np.zeros_like(bg)
    for j in range(ny):
        for i in range(nx):
            color = tuple(int(x) for x in lut[int(np.clip(g[j, i] * 255, 0, 255))])
            cv2.fillConvexPoly(heat, cells[j][i], color, cv2.LINE_AA)

    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    a = qmask[..., None]
    base = bg.astype(np.float32) * (1 - a) + gray3 * a
    ah = (alpha * qmask)[..., None]
    out = (base * (1 - ah) + heat.astype(np.float32) * ah).astype(np.uint8)

    lines = out.copy()
    for j in range(ny):
        for i in range(nx):
            cv2.polylines(lines, [cells[j][i]], True, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.addWeighted(lines, 0.18, out, 0.82, 0)


def make_heatmap(
    video, start: float = 0.0, duration: float | None = None, stride: int = 3,
    conf: float = 0.5, model: str = "medium", nx: int = 12, ny: int = 8,
    power: float = 0.6, alpha: float = 0.62, output=None, show: bool = True,
) -> str:
    """Render the zonal court heatmap for a clip and save it (optionally show it)."""
    quad = calibration.court(video)
    if quad is None:
        raise SystemExit("no court calibration — run `padel-vision court adjust <video>` first")
    roi = calibration.roi(video)
    zone = sv.PolygonZone(polygon=roi) if roi is not None else None
    h_inv = cv2.getPerspectiveTransform(quad, _UNIT)
    h_fwd = cv2.getPerspectiveTransform(_UNIT, quad)

    info = video_info(video)
    fps = max(1, round(info.fps))
    start_f = int(start * fps)
    n_frames = (info.total_frames - start_f) if duration is None else int(duration * fps)

    cfg = Config().detector
    cfg.confidence = conf
    cfg.model = model
    detector = build_detector(cfg)

    grid = np.zeros((ny, nx), np.float32)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    for k in tqdm(range(n_frames), desc="heatmap"):
        ok, frame = cap.read()
        if not ok:
            break
        if k % stride:
            continue
        dets = detector.detect(frame)
        if zone is not None:
            dets = dets[zone.trigger(dets)]
        for x1, _y1, x2, y2 in dets.xyxy:
            uv = cv2.perspectiveTransform(
                np.float32([[(x1 + x2) / 2, y2]]).reshape(-1, 1, 2), h_inv
            ).reshape(-1)
            if 0 <= uv[0] < 1 and 0 <= uv[1] < 1:
                grid[int(uv[1] * ny), int(uv[0] * nx)] += 1
    cap.release()

    out_img = _render(grab_frame(video, start_f), grid, quad, h_fwd, nx, ny, power, alpha)
    output = Path(output) if output else Path("data/processed") / f"{Path(video).stem}_heatmap.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), out_img)
    print(f"heatmap -> {output}")

    if show:
        try:
            cv2.imshow("padel-vision heatmap  (any key to close)", out_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # no display — the image is saved regardless
    return str(output)
