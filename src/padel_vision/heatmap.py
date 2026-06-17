"""Single-frame zonal court heatmap preview.

Detect players on one frame, seed a dummy movement grid around their foot points,
and render a broadcast-style green->red heatmap with detections and foreground
segmentation. Needs ``court adjust`` first; uses the ROI if one is saved.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from . import calibration
from .config import Config
from .detection.detector import build_detector
from .rings import DEFAULT_RING
from .track import ForegroundMatte, FullOverlayRenderer
from .video.io import grab_frame, video_info

_UNIT = np.float32([(0, 0), (1, 0), (1, 1), (0, 1)])
_GRID_ALPHA = 0.62


def _rdylgn_r_lut() -> np.ndarray:
    """ColorBrewer RdYlGn_r-style BGR LUT, matching the notebook section 9 palette."""
    anchors = np.array(
        [
            (0, 104, 55),
            (26, 152, 80),
            (102, 189, 99),
            (166, 217, 106),
            (217, 239, 139),
            (255, 255, 191),
            (254, 224, 139),
            (253, 174, 97),
            (244, 109, 67),
            (215, 48, 39),
            (165, 0, 38),
        ],
        dtype=np.float32,
    )
    x = np.linspace(0, len(anchors) - 1, 256)
    lo = np.floor(x).astype(int)
    hi = np.clip(lo + 1, 0, len(anchors) - 1)
    t = (x - lo)[:, None]
    rgb = anchors[lo] * (1 - t) + anchors[hi] * t
    return rgb[:, ::-1].astype(np.uint8)


def _cell(Hc, i, j, nx, ny) -> np.ndarray:
    u0, u1, v0, v1 = i / nx, (i + 1) / nx, j / ny, (j + 1) / ny
    corners = np.float32([(u0, v0), (u1, v0), (u1, v1), (u0, v1)]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(corners, Hc).reshape(-1, 2).astype(np.int32)


def _render(bg, grid, quad, Hc, nx, ny, power, alpha) -> np.ndarray:
    h, w = bg.shape[:2]
    cells = [[_cell(Hc, i, j, nx, ny) for i in range(nx)] for j in range(ny)]
    qmask = np.zeros((h, w), np.float32)
    cv2.fillConvexPoly(qmask, quad.astype(np.int32), 1.0)

    val = cv2.GaussianBlur(grid, (0, 0), 0.8)
    val = val / max(float(val.max()), 1.0)
    val = np.power(val, power)
    lut = _rdylgn_r_lut()
    heat = np.zeros_like(bg)
    for j in range(ny):
        for i in range(nx):
            color = tuple(int(x) for x in lut[int(np.clip(val[j, i] * 255, 0, 255))])
            cv2.fillConvexPoly(heat, cells[j][i], color, cv2.LINE_AA)

    gray3 = cv2.cvtColor(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    area = (alpha * qmask)[..., None]
    base = bg.astype(np.float32) * (1 - area) + gray3.astype(np.float32) * area
    heat_area = (_GRID_ALPHA * alpha * qmask)[..., None]
    out = (base * (1 - heat_area) + heat.astype(np.float32) * heat_area).astype(np.uint8)

    gridlines = np.zeros_like(bg)
    for row in cells:
        for cell in row:
            cv2.polylines(gridlines, [cell], True, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.addWeighted(out, 1.0, gridlines, 0.22 * alpha, 0)


def _dummy_grid(dets: sv.Detections, h_inv, nx: int, ny: int) -> np.ndarray:
    """Create fake per-player occupancy around the detected foot points."""
    grid = np.zeros((ny, nx), np.float32)
    yy, xx = np.mgrid[0:ny, 0:nx]
    for idx, (x1, _y1, x2, y2) in enumerate(dets.xyxy):
        uv = cv2.perspectiveTransform(
            np.float32([[(x1 + x2) / 2, y2]]).reshape(-1, 1, 2), h_inv
        ).reshape(-1)
        if not (0 <= uv[0] < 1 and 0 <= uv[1] < 1):
            continue
        cx = uv[0] * nx - 0.5
        cy = uv[1] * ny - 0.5
        weight = 1.0 + 0.25 * idx
        grid += weight * np.exp(-(((xx - cx) ** 2) / 2.2 + ((yy - cy) ** 2) / 1.6))
    return grid


def _with_dummy_track_ids(dets: sv.Detections) -> sv.Detections:
    """Give one-frame detections stable-looking IDs so rings match track players."""
    if len(dets) == 0:
        return dets
    dets.tracker_id = np.arange(1, len(dets) + 1, dtype=int)
    return dets


def make_heatmap(
    video, start: float = 0.0, duration: float | None = None, stride: int = 3,
    conf: float = 0.5, model: str = "medium", nx: int = 12, ny: int = 8,
    power: float = 0.4, alpha: float = 1.0, output=None, show: bool = True,
    frame: int | None = None, foreground: bool = True,
    foreground_model: str = "yolo11n-seg.pt", trail: bool = False, labels: bool = False,
) -> str:
    """Render a single-frame heatmap preview and save it (optionally show it).

    ``duration`` and ``stride`` are accepted for CLI compatibility with the older
    accumulation command, but this preview mode renders exactly one frame.
    """
    quad = calibration.court(video)
    if quad is None:
        raise SystemExit("no court calibration — run `padel-vision court adjust <video>` first")
    roi = calibration.roi(video)
    zone = sv.PolygonZone(polygon=roi) if roi is not None else None
    matte_polygon = roi if roi is not None else quad.astype(np.int32)
    h_inv = cv2.getPerspectiveTransform(quad, _UNIT)
    h_fwd = cv2.getPerspectiveTransform(_UNIT, quad)

    info = video_info(video)
    fps = max(1, round(info.fps))
    frame_idx = int(frame) if frame is not None else int(start * fps)

    cfg = Config().detector
    cfg.confidence = conf
    cfg.model = model
    detector = build_detector(cfg)

    bg = grab_frame(video, frame_idx)
    dets = detector.detect(bg)
    if zone is not None:
        dets = dets[zone.trigger(dets)]

    grid = _dummy_grid(dets, h_inv, nx, ny)
    heatmap_base = _render(bg, grid, quad, h_fwd, nx, ny, power, alpha)

    ring_params = dict(DEFAULT_RING)
    ring_params.update(calibration.ring(video) or {})
    dets = _with_dummy_track_ids(dets)
    matte = ForegroundMatte(matte_polygon, foreground_model)(bg) if foreground else None
    renderer = FullOverlayRenderer(
        bg.shape[0], matte_polygon, ring_params, fps, trail=trail, labels=labels
    )
    out_img = renderer.render(
        bg, dets, frame_idx / fps, live_fps=0.0, model=model, matte=matte, base=heatmap_base
    )

    output = Path(output) if output else Path("data/processed") / f"{Path(video).stem}_heatmap.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), out_img)
    print(f"heatmap preview -> {output} ({len(dets)} detections on frame {frame_idx})")

    if show:
        try:
            cv2.imshow("padel-vision heatmap  (any key to close)", out_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # no display — the image is saved regardless
    return str(output)
