"""Interactive guided workflow for configuring a padel clip."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import calibration
from .court import CORNER_NAMES
from .final_render import build_detection_cache, render_final_cut
from .pickers import pick_points
from .video.io import grab_frame


@dataclass
class GuideConfig:
    video: Path
    frame: int = 300
    court_map: Path = Path("data/court_map.png")
    cache_output: Path = Path("data/processed/detect_cache.pkl")
    final_output: Path = Path("data/processed/padel_final.mp4")
    model: str = "nano"
    stride: int = 2
    conf: float = 0.3


def _prompt(text: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default not in (None, "") else ""
    value = input(f"{text}{suffix}: ").strip()
    return value or (default or "")


def _prompt_bool(text: str, default: bool = True) -> bool:
    marker = "Y/n" if default else "y/N"
    while True:
        value = input(f"{text} [{marker}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please answer yes or no.")


def _confirm_or_redo(path: Path, label: str) -> bool:
    if path.exists():
        print(f"Found existing {label}: {path}")
        return _prompt_bool(f"Reuse this {label}?", default=True)
    return False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("\nRunning:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _pick_roi(cfg: GuideConfig) -> Path | None:
    frame = grab_frame(cfg.video, cfg.frame)
    pts = pick_points(
        frame,
        n=None,
        title="ROI - click court/play area polygon, then f to finish",
    )
    if pts is None:
        print("ROI cancelled; nothing saved.")
        return None
    path = calibration.save_roi(cfg.video, pts)
    print(f"Saved ROI ({len(pts)} points) -> {path}")
    return path


def _pick_court(cfg: GuideConfig) -> Path | None:
    frame = grab_frame(cfg.video, cfg.frame)
    pts = pick_points(
        frame,
        n=4,
        labels=CORNER_NAMES,
        title="Court - click TL, TR, BR, BL",
    )
    if pts is None:
        print("Court calibration cancelled; nothing saved.")
        return None
    path = calibration.save_court(cfg.video, pts)
    print(f"Saved court corners -> {path}")
    return path


def _bev_path(video: Path) -> Path:
    return Path("data/calibration") / f"{video.stem}_bev_points.json"


def _pick_bev(cfg: GuideConfig) -> None:
    script = _repo_root() / "scripts" / "pick_bev_points.py"
    cmd = [
        sys.executable,
        str(script),
        "--video",
        str(cfg.video),
        "--frame",
        str(cfg.frame),
        "--map",
        str(cfg.court_map),
        "--out",
        str(_bev_path(cfg.video)),
    ]
    _run(cmd, cwd=_repo_root())


def _has_grid_roi(video: Path) -> bool:
    path = _bev_path(video)
    if not path.exists():
        return False
    try:
        return len(json.loads(path.read_text()).get("grid_map_quad", [])) == 4
    except ValueError:
        return False


def _pick_grid_roi(cfg: GuideConfig) -> None:
    script = _repo_root() / "scripts" / "pick_grid_roi.py"
    cmd = [
        sys.executable,
        str(script),
        "--map",
        str(cfg.court_map),
        "--bev-json",
        str(_bev_path(cfg.video)),
    ]
    _run(cmd, cwd=_repo_root())


def guide(
    video: str | None = None,
    frame: int = 300,
    court_map: str = "data/court_map.jpg",
    stride: int = 2,
    model: str = "nano",
    conf: float = 0.3,
) -> None:
    """Guide a user through clip calibration and final renderer layout.

    The guide is intentionally resumable: when calibration files already exist,
    it asks whether to reuse them before opening any interactive picker.
    """
    print("\nPadel Vision guided setup")
    print("=========================")
    print("You can press Ctrl+C at any time; completed calibration files are kept.\n")

    video_value = video or _prompt("Source video", "data/raw/padel_clip.mp4")
    cfg = GuideConfig(
        video=Path(video_value),
        frame=int(frame),
        court_map=Path(court_map),
        stride=int(stride),
        model=model,
        conf=float(conf),
    )

    if not cfg.video.exists():
        raise FileNotFoundError(f"Video not found: {cfg.video}")
    if not cfg.court_map.exists():
        print(f"Court map not found: {cfg.court_map}")
        replacement = _prompt("Court map image", str(cfg.court_map))
        cfg.court_map = Path(replacement)
        if not cfg.court_map.exists():
            raise FileNotFoundError(f"Court map not found: {cfg.court_map}")

    frame_text = _prompt("Calibration frame", str(cfg.frame))
    cfg.frame = int(frame_text)

    print("\nStep 1/5: detection ROI")
    cal_path = calibration.path_for(cfg.video)
    cal = calibration.load(cfg.video)
    if "roi" in cal and _confirm_or_redo(cal_path, "ROI calibration"):
        print("Keeping existing ROI.")
    elif _prompt_bool("Open ROI picker now?", default=True):
        _pick_roi(cfg)

    print("\nStep 2/5: court corners")
    cal = calibration.load(cfg.video)
    if "court" in cal and _confirm_or_redo(cal_path, "court calibration"):
        print("Keeping existing court corners.")
    elif _prompt_bool("Open court corner picker now?", default=True):
        _pick_court(cfg)

    print("\nStep 3/6: bird's-eye-view point pairs")
    bev = _bev_path(cfg.video)
    if bev.exists() and _confirm_or_redo(bev, "BEV point pairs"):
        print("Keeping existing BEV point pairs.")
    elif _prompt_bool("Open BEV point-pair picker now?", default=True):
        _pick_bev(cfg)

    print("\nStep 4/6: heatmap grid ROI on 2D map")
    if _has_grid_roi(cfg.video) and _prompt_bool("Reuse existing heatmap grid ROI?", default=True):
        print("Keeping existing heatmap grid ROI.")
    elif _prompt_bool("Open heatmap grid ROI picker now?", default=True):
        _pick_grid_roi(cfg)

    print("\nStep 5/6: notebook detection cache")
    cfg.stride = int(_prompt("Frame stride", str(cfg.stride)))
    cfg.conf = float(_prompt("RF-DETR segmentation confidence", str(cfg.conf)))
    cfg.model = _prompt("RF-DETR segmentation model", cfg.model)
    if cfg.cache_output.exists() and _confirm_or_redo(cfg.cache_output, "detection cache"):
        print("Keeping existing detection cache.")
    elif _prompt_bool("Build notebook detection cache now?", default=True):
        build_detection_cache(
            cfg.video,
            output=cfg.cache_output,
            stride=cfg.stride,
            model=cfg.model,
            conf=cfg.conf,
        )

    print("\nStep 6/6: notebook final layout")
    show = _prompt_bool("Show live preview while rendering?", default=False)
    if _prompt_bool("Render final notebook-style video now?", default=True):
        render_final_cut(
            cfg.video,
            cache=cfg.cache_output,
            bev_json=_bev_path(cfg.video),
            court_map=cfg.court_map,
            output=cfg.final_output,
            show=show,
        )

    print("\nDone. Current configuration:")
    print(f"  video: {cfg.video}")
    print(f"  calibration: {calibration.path_for(cfg.video)}")
    print(f"  bev pairs: {_bev_path(cfg.video)}")
    print(f"  heatmap grid ROI: {_bev_path(cfg.video)} -> grid_map_quad")
    print(f"  detection cache: {cfg.cache_output}")
    print(f"  final video: {cfg.final_output}")
