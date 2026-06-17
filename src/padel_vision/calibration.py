"""Per-clip calibration store.

Each video gets one structured JSON file under ``data/calibration/<stem>.json``
holding its **ROI** (a detection-filter polygon) and **court** (the 4 corners used
for the heatmap homography), e.g.::

    {
      "video": "match.mp4",
      "roi":   [[x, y], [x, y], ...],
      "court": {"TL": [x, y], "TR": [x, y], "BR": [x, y], "BL": [x, y]}
    }

The CLI writes it (``padel-vision roi adjust`` / ``court adjust``) and later
commands (``detect players``, …) load it automatically.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .court import CORNER_NAMES

DEFAULT_DIR = Path("data/calibration")


def path_for(video: str | Path, base: str | Path = DEFAULT_DIR) -> Path:
    """The calibration file path for a clip (keyed by its filename stem)."""
    return Path(base) / f"{Path(video).stem}.json"


def load(video: str | Path, base: str | Path = DEFAULT_DIR) -> dict:
    """Return the clip's calibration dict (``{}`` if none saved yet)."""
    p = path_for(video, base)
    return json.loads(p.read_text()) if p.exists() else {}


def _save(video: str | Path, data: dict, base: str | Path) -> Path:
    p = path_for(video, base)
    p.parent.mkdir(parents=True, exist_ok=True)
    data["video"] = Path(video).name
    p.write_text(json.dumps(data, indent=2) + "\n")
    return p


def save_roi(video: str | Path, points, base: str | Path = DEFAULT_DIR) -> Path:
    """Save the detection-filter polygon (list of ``(x, y)``)."""
    data = load(video, base)
    data["roi"] = [[int(x), int(y)] for x, y in points]
    return _save(video, data, base)


def save_court(video: str | Path, corners, base: str | Path = DEFAULT_DIR) -> Path:
    """Save the 4 court corners (in TL, TR, BR, BL order)."""
    data = load(video, base)
    data["court"] = {n: [int(x), int(y)] for n, (x, y) in zip(CORNER_NAMES, corners, strict=True)}
    return _save(video, data, base)


def save_ring(video: str | Path, params: dict, base: str | Path = DEFAULT_DIR) -> Path:
    """Save the AR ground-ring style params (radius, tilt, persp, ...)."""
    data = load(video, base)
    data["ring"] = dict(params)
    return _save(video, data, base)


def roi(video: str | Path, base: str | Path = DEFAULT_DIR) -> np.ndarray | None:
    """The ROI polygon as an ``(N, 2)`` int32 array, or ``None`` if unset."""
    pts = load(video, base).get("roi")
    return np.array(pts, dtype=np.int32) if pts else None


def court(video: str | Path, base: str | Path = DEFAULT_DIR) -> np.ndarray | None:
    """The court corners as a ``(4, 2)`` float32 array (TL, TR, BR, BL), or ``None``."""
    c = load(video, base).get("court")
    return np.float32([c[n] for n in CORNER_NAMES]) if c else None


def ring(video: str | Path, base: str | Path = DEFAULT_DIR) -> dict | None:
    """The saved AR ground-ring style params, or ``None`` if unset."""
    return load(video, base).get("ring")
