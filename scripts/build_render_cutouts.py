"""Build cropped RGBA player cutouts from an existing Remotion export.

This is much faster than rerunning detection. It reads:

    renderer/public/render/frames.json
    renderer/public/render/frames/*.jpg
    renderer/public/render/masks/*.png

and writes:

    renderer/public/render/cutouts/*.png

Then it updates each player record with ``cutout`` and ``cutoutBox``.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def save_cutout(
    frame: np.ndarray,
    mask: np.ndarray,
    bbox: list[float],
    path: Path,
    padding: int,
) -> list[int] | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    left = max(0, int(math.floor(x1)) - padding)
    top = max(0, int(math.floor(y1)) - padding)
    right = min(w, int(math.ceil(x2)) + padding)
    bottom = min(h, int(math.ceil(y2)) + padding)
    if right <= left or bottom <= top:
        return None

    rgb = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2RGB)
    alpha = mask[top:bottom, left:right]
    if alpha.ndim == 3:
        alpha = alpha[:, :, 0]
    rgba = np.dstack([rgb, alpha])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    return [left, top, right - left, bottom - top]


def build(args: argparse.Namespace) -> Path:
    render_dir = Path(args.render_dir)
    payload_path = render_dir / "frames.json"
    payload = json.loads(payload_path.read_text())
    cutouts_dir = render_dir / "cutouts"

    if args.clean and cutouts_dir.exists():
        shutil.rmtree(cutouts_dir)
    cutouts_dir.mkdir(parents=True, exist_ok=True)

    made = 0
    for frame_payload in tqdm(payload["frames"], desc="build player cutouts"):
        image_rel = frame_payload.get("image")
        if not image_rel:
            continue

        frame_path = render_dir.parent / image_rel
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        for player in frame_payload.get("players", []):
            mask_rel = player.get("mask")
            if not mask_rel:
                player["cutout"] = None
                player["cutoutBox"] = None
                continue

            mask_path = render_dir.parent / mask_rel
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                player["cutout"] = None
                player["cutoutBox"] = None
                continue

            source_frame = frame_payload.get("sourceFrame", frame_payload["frame"])
            player_id = str(player["id"]).lower()
            cutout_name = f"frame_{int(source_frame):06d}_{player_id}.png"
            cutout_box = save_cutout(
                frame,
                mask,
                player["bbox"],
                cutouts_dir / cutout_name,
                args.padding,
            )
            player["cutout"] = f"render/cutouts/{cutout_name}" if cutout_box else None
            player["cutoutBox"] = cutout_box
            made += 1 if cutout_box else 0

    payload_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Built {made} cutouts -> {cutouts_dir}")
    print(f"Updated {payload_path}")
    return payload_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", default="renderer/public/render")
    parser.add_argument("--padding", type=int, default=10)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
