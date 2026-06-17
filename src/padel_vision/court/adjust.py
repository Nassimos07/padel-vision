"""Interactive court-corner picker.

Open a frame from a clip, click the four corners of the playing surface
(TL, TR, BR, BL), and save them to a small text file that the rest of the
project (and the tutorial notebooks) reads back as the court homography.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Order matters everywhere downstream: top-left, top-right, bottom-right, bottom-left.
CORNER_NAMES: tuple[str, ...] = ("TL", "TR", "BR", "BL")

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def save_corners(corners, path: str | Path = "court_corners.txt") -> Path:
    """Write four ``(x, y)`` corners to ``path`` in TL, TR, BR, BL order."""
    corners = list(corners)
    if len(corners) != 4:
        raise ValueError(f"expected 4 corners, got {len(corners)}")
    path = Path(path)
    lines = ["# Court corners (image pixels) — order: TL, TR, BR, BL"]
    for name, (x, y) in zip(CORNER_NAMES, corners, strict=True):
        lines.append(f"{name} {int(round(x))} {int(round(y))}")
    path.write_text("\n".join(lines) + "\n")
    return path


def load_corners(path: str | Path = "court_corners.txt") -> np.ndarray:
    """Read corners back as a ``(4, 2)`` float32 array in TL, TR, BR, BL order."""
    found: dict[str, tuple[float, float]] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, x, y = line.split()
            found[name] = (float(x), float(y))
    return np.float32([found[name] for name in CORNER_NAMES])


def _read_frame(video: str | Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise FileNotFoundError(f"could not read frame {frame_idx} from {video!r}")
    return frame


def _draw(img, pts):
    out = img.copy()
    poly = np.array(pts, np.int32)
    if len(pts) >= 2:
        cv2.polylines(out, [poly], len(pts) == 4, (0, 255, 255), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        cv2.circle(out, (x, y), 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(out, CORNER_NAMES[i], (x + 10, y - 10), _FONT, 0.8, (0, 255, 255), 2)
    hint = (f"click {CORNER_NAMES[len(pts)]}" if len(pts) < 4
            else "press 's' save  ·  'r' reset  ·  'q' quit")
    cv2.putText(out, hint, (20, 44), _FONT, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def adjust(
    video: str | Path, output: str | Path = "court_corners.txt", frame: int = 300
) -> str | None:
    """Pick the 4 court corners on a frame and save them.

    Controls: **left-click** place a corner · **u** undo · **r** reset · **s** save · **q** quit.
    Needs a desktop display (OpenCV window) — not Colab/headless. Returns the
    output path on save, or ``None`` if cancelled.
    """
    img = _read_frame(video, int(frame))
    pts: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):  # noqa: ANN001  (OpenCV callback signature)
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((int(x), int(y)))

    win = "padel-vision court adjust"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        cv2.imshow(win, _draw(img, pts))
        key = cv2.waitKey(20) & 0xFF
        if key == ord("u") and pts:
            pts.pop()
        elif key == ord("r"):
            pts.clear()
        elif key in (ord("q"), 27):  # q or Esc
            cv2.destroyWindow(win)
            print("cancelled — nothing saved")
            return None
        elif key == ord("s") and len(pts) == 4:
            out = save_corners(pts, output)
            cv2.destroyWindow(win)
            print(f"saved 4 corners -> {out}")
            return str(out)
