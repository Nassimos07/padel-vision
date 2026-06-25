"""Pick the heatmap grid rectangle on the 2D court map.

Click the four grid ROI corners on the flat map in this order:

    TL, TR, BR, BL

The points are stored in the existing BEV JSON under ``grid_map_quad``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
LABELS = ("TL", "TR", "BR", "BL")
FOOTER = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="data/court_map.jpg", help="flat 2D court image")
    parser.add_argument("--bev-json", required=True, help="BEV calibration JSON to update")
    parser.add_argument("--height", type=int, default=760, help="display map height")
    args = parser.parse_args()

    bev_path = Path(args.bev_json)
    if not bev_path.exists():
        raise SystemExit(f"BEV JSON not found: {bev_path}")

    court_map = cv2.imread(args.map)
    if court_map is None:
        raise SystemExit(f"map image not found: {args.map}")

    data = json.loads(bev_path.read_text())
    pts: list[list[float]] = [list(p) for p in data.get("grid_map_quad", [])]
    if pts:
        print(f"resuming with {len(pts)} heatmap grid points from {bev_path}")

    mh, mw = court_map.shape[:2]
    scale = args.height / mh
    display_w = int(mw * scale)
    map_display = cv2.resize(court_map, (display_w, args.height))

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN or y >= args.height or len(pts) >= 4:
            return
        pts.append([float(x / scale), float(y / scale)])

    def draw():
        canvas = np.zeros((args.height + FOOTER, display_w, 3), np.uint8)
        canvas[: args.height, :display_w] = map_display
        disp_pts = [(int(x * scale), int(y * scale)) for x, y in pts]
        if len(disp_pts) >= 2:
            cv2.polylines(
                canvas,
                [np.array(disp_pts, np.int32)],
                len(disp_pts) == 4,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        for i, (x, y) in enumerate(disp_pts):
            cv2.circle(canvas, (x, y), 7, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(canvas, LABELS[i], (x + 10, y - 10), FONT, 0.7, (0, 255, 255), 2)
        next_label = LABELS[len(pts)] if len(pts) < 4 else "save"
        status = (
            f"grid ROI: {len(pts)}/4   next: {next_label}   |   "
            "u=undo  r=reset  s=save  q=quit"
        )
        cv2.rectangle(canvas, (0, args.height), (display_w, args.height + FOOTER), (20, 20, 20), -1)
        cv2.putText(
            canvas,
            status,
            (12, args.height + 27),
            FONT,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return canvas

    win = "Heatmap grid ROI - TL TR BR BL"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win, draw())
    cv2.waitKey(1)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        cv2.imshow(win, draw())
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("u") and pts:
            pts.pop()
        if key == ord("r"):
            pts.clear()
        if key == ord("s"):
            if len(pts) != 4:
                print("pick exactly 4 points first: TL, TR, BR, BL")
                continue
            data["grid_map_quad"] = pts
            data["grid_map_quad_order"] = list(LABELS)
            data["map_path"] = args.map
            data["map_size"] = [mw, mh]
            bev_path.write_text(json.dumps(data, indent=2) + "\n")
            print(f"saved heatmap grid ROI -> {bev_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
