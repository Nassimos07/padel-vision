"""Interactive OpenCV point pickers (need a desktop display — not Colab/headless).

``pick_points`` handles both calibration tasks:
  * a **fixed** number of labelled points (the 4 court corners), and
  * a **variable** polygon (the detection ROI), finished with a key.
"""

from __future__ import annotations

import cv2
import numpy as np

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WIN = "padel-vision"  # short ASCII window name (robust across GUI backends / WSLg)


def pick_points(frame, n: int | None = None, labels=None, title: str = "pick points"):
    """Click points on ``frame``; return ``list[(x, y)]`` or ``None`` if cancelled.

    Args:
        n: fixed number of points (auto-closes the shape), or ``None`` for a
           variable-length polygon finished with ``f``/``Enter`` (needs >= 3 points).
        labels: optional per-point labels (e.g. corner names).
        title: a short ASCII instruction drawn in the window.

    Controls: click add | ``u`` undo | ``r`` reset | ``f``/Enter finish | ``q``/Esc cancel.
    """
    pts: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):  # noqa: ANN001  (OpenCV callback)
        if event == cv2.EVENT_LBUTTONDOWN and (n is None or len(pts) < n):
            pts.append((int(x), int(y)))

    cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_WIN, 1280, 720)
    # Realize the window before attaching the mouse callback — under WSLg/Qt the
    # window is created asynchronously, so setMouseCallback right after namedWindow
    # can hit a NULL window handle.
    cv2.imshow(_WIN, frame)
    cv2.waitKey(1)
    cv2.setMouseCallback(_WIN, on_mouse)

    def _complete() -> bool:
        return len(pts) == n if n is not None else len(pts) >= 3

    while True:
        disp = frame.copy()
        if len(pts) >= 2:
            poly = np.array(pts, np.int32)
            cv2.polylines(disp, [poly], _complete(), (0, 255, 255), 2, cv2.LINE_AA)
        for i, (x, y) in enumerate(pts):
            cv2.circle(disp, (x, y), 6, (0, 0, 255), -1, cv2.LINE_AA)
            lbl = labels[i] if labels and i < len(labels) else str(i + 1)
            cv2.putText(disp, lbl, (x + 10, y - 10), _FONT, 0.7, (0, 255, 255), 2)
        if n is not None and len(pts) < n:
            hint = f"click {labels[len(pts)] if labels else len(pts) + 1}  ({len(pts)}/{n})"
        elif n is None and not _complete():
            hint = f"{len(pts)} pts | click to add | 'f' to finish (>=3)"
        else:
            hint = "'f' finish | 'u' undo | 'r' reset | 'q' quit"
        cv2.putText(disp, title, (20, 36), _FONT, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, hint, (20, 70), _FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(_WIN, disp)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("u") and pts:
            pts.pop()
        elif key == ord("r"):
            pts.clear()
        elif key in (ord("q"), 27):
            cv2.destroyWindow(_WIN)
            return None
        elif key in (ord("f"), ord("s"), 13) and _complete():  # f / s / Enter
            cv2.destroyWindow(_WIN)
            return pts
