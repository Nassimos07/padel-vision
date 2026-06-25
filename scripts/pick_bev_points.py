"""Collect frame ↔ 2D-map point couples for a bird's-eye-view homography.

Shows the chosen video frame (left) and the flat 2D court map (right) side by side.
Click a point on the **frame**, then the **same** point on the **map** — that's one
couple. Collect ~6-10 well-spread couples (court corners, service-line ends, the T
junctions, net posts) and save them; later we fit a homography from these to warp the
broadcast view into a top-down map and project the players onto it.

Run::

    python scripts/pick_bev_points.py \
        --video data/raw/padel_clip.mp4 --frame 300 --map data/court_map.png

Controls:  click FRAME then MAP to add a couple · u = undo last · r = reset all
           s = save · q / Esc = quit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

PALETTE = [
    (0, 229, 255), (174, 61, 255), (63, 210, 255), (107, 255, 124),
    (255, 128, 0), (0, 165, 255), (255, 255, 0), (180, 105, 255),
    (120, 255, 200), (255, 90, 160),
]
FONT = cv2.FONT_HERSHEY_SIMPLEX
FOOTER = 38


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default="data/raw/padel_clip.mp4")
    ap.add_argument("--frame", type=int, default=300, help="frame index to calibrate on")
    ap.add_argument("--map", default="data/court_map.png", help="flat 2D court image")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: calibration/<stem>_bev_points.json)")
    ap.add_argument("--height", type=int, default=680, help="display height of each pane")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else (
        Path("data/calibration") / f"{Path(args.video).stem}_bev_points.json"
    )

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {args.frame} from {args.video}")

    court_map = cv2.imread(args.map)
    if court_map is None:
        raise SystemExit(f"map image not found: {args.map}  (save the 2D court image there first)")

    # resume a previous session if one exists
    pairs: list[dict] = []
    if out_path.exists():
        try:
            pairs = json.loads(out_path.read_text()).get("pairs", [])
            print(f"resuming with {len(pairs)} existing couples from {out_path}")
        except json.JSONDecodeError:
            pass

    fh, fw = frame.shape[:2]
    mh, mw = court_map.shape[:2]
    sf, sm = args.height / fh, args.height / mh
    fw_d, mw_d = int(fw * sf), int(mw * sm)
    frame_d = cv2.resize(frame, (fw_d, args.height))
    map_d = cv2.resize(court_map, (mw_d, args.height))
    map_x0 = fw_d + 40                       # x where the map pane starts on the canvas
    canvas_w = map_x0 + mw_d

    state = {"pending": None}                # frame point (original coords) awaiting its map point

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN or y >= args.height:
            return
        on_frame, on_map = x < fw_d, x >= map_x0
        if on_frame:
            state["pending"] = (x / sf, y / sf)                  # (re)pick the frame point
        elif on_map and state["pending"] is not None:
            mx, my = (x - map_x0) / sm, y / sm
            pairs.append({"frame": [float(state["pending"][0]), float(state["pending"][1])],
                          "map": [float(mx), float(my)]})
            state["pending"] = None

    def draw():
        canvas = np.zeros((args.height + FOOTER, canvas_w, 3), np.uint8)
        canvas[:args.height, :fw_d] = frame_d
        canvas[:args.height, map_x0:map_x0 + mw_d] = map_d
        for i, p in enumerate(pairs):
            c = PALETTE[i % len(PALETTE)]
            fx, fy = int(p["frame"][0] * sf), int(p["frame"][1] * sf)
            mx, my = int(p["map"][0] * sm + map_x0), int(p["map"][1] * sm)
            for px, py in ((fx, fy), (mx, my)):
                cv2.circle(canvas, (px, py), 6, c, -1, cv2.LINE_AA)
                cv2.circle(canvas, (px, py), 6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(canvas, str(i + 1), (px + 8, py - 8), FONT, 0.6, c, 2, cv2.LINE_AA)
        if state["pending"] is not None:
            fx, fy = int(state["pending"][0] * sf), int(state["pending"][1] * sf)
            cv2.circle(canvas, (fx, fy), 8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "now click the SAME point on the map  ->", (fx + 12, fy),
                        FONT, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        nxt = "MAP point" if state["pending"] is not None else "FRAME point"
        status = f"couples: {len(pairs)}   next: click {nxt}   |   u=undo  r=reset  s=save  q=quit"
        cv2.rectangle(canvas, (0, args.height), (canvas_w, args.height + FOOTER), (20, 20, 20), -1)
        cv2.putText(canvas, status, (12, args.height + 25), FONT, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    win = "BEV point picker"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win, draw())
    cv2.waitKey(1)                           # realize the window before the callback (WSLg/Qt)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        cv2.imshow(win, draw())
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("u") and pairs:
            pairs.pop()
        if k == ord("r"):
            pairs.clear()
            state["pending"] = None
        if k == ord("s"):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({
                "video": Path(args.video).name, "frame_idx": args.frame,
                "map_path": args.map, "map_size": [mw, mh], "frame_size": [fw, fh],
                "pairs": pairs,
            }, indent=2) + "\n")
            print(f"saved {len(pairs)} couples -> {out_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
