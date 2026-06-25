"""Export tracking data for the Remotion renderer.

This is the bridge between the Python CV pipeline and the React/SVG renderer:

    python scripts/export_remotion_data.py data/raw/padel_clip.mp4 --seconds 5 --stride 2

Output lands in ``renderer/public/render/``:

    frames.json
    frames/frame_000000.jpg
    masks/frame_000000_p1.png

The script intentionally uses one segmentation model for both boxes and masks.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

from padel_vision import calibration
from padel_vision.court import CORNER_NAMES

PALETTE = ["#ff3d8b", "#ffb020", "#27e08a", "#2e9bff"]
UNIT = np.float32([(0, 0), (1, 0), (1, 1), (0, 1)])


class TrackStabilizer:
    """EMA-smooth boxes and coast briefly through dropped detections."""

    def __init__(self, smoothing: float = 0.35, hold_frames: int = 30) -> None:
        self.alpha = float(smoothing)
        self.hold = int(hold_frames)
        self.state: dict[int, dict] = {}

    def update(self, detections: sv.Detections, frame_idx: int) -> sv.Detections:
        if detections.tracker_id is not None:
            confs = (
                detections.confidence
                if detections.confidence is not None
                else np.ones(len(detections))
            )
            for xyxy, tracker_id, conf in zip(
                detections.xyxy, detections.tracker_id, confs, strict=False
            ):
                tracker_id = int(tracker_id)
                xyxy = np.asarray(xyxy, dtype=float)
                current = self.state.get(tracker_id)
                if current is None:
                    current = {"xyxy": xyxy.copy()}
                else:
                    current["xyxy"] = self.alpha * xyxy + (1 - self.alpha) * current["xyxy"]
                current["conf"] = float(conf)
                current["last_seen"] = int(frame_idx)
                self.state[tracker_id] = current

        ids, boxes, confs = [], [], []
        for tracker_id, current in list(self.state.items()):
            if frame_idx - current["last_seen"] > self.hold:
                del self.state[tracker_id]
                continue
            ids.append(tracker_id)
            boxes.append(current["xyxy"])
            confs.append(current["conf"])

        if not ids:
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=np.array(boxes, dtype=float),
            confidence=np.array(confs, dtype=float),
            class_id=np.zeros(len(ids), dtype=int),
            tracker_id=np.array(ids, dtype=int),
        )


class StableIdentityMapper:
    """Remap raw ByteTrack IDs into stable visual IDs."""

    def __init__(
        self,
        homography_inv,
        max_identities: int = 4,
        max_age: int = 45,
        max_cost: float = 0.38,
        fallback_cost: float = 0.85,
    ) -> None:
        self.hi = homography_inv
        self.max_identities = int(max_identities)
        self.max_age = int(max_age)
        self.max_cost = float(max_cost)
        self.fallback_cost = float(fallback_cost)
        self.state: dict[int, dict] = {}
        self.raw_to_stable: dict[int, int] = {}
        self.slot_sides: dict[int, int] = {}

    def _foot_uv(self, xyxy) -> np.ndarray:
        x1, _y1, x2, y2 = xyxy
        pt = np.float32([[(x1 + x2) / 2, y2]]).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pt, self.hi).reshape(-1)

    def _predict(self, stable_id: int, frame_idx: int) -> np.ndarray:
        current = self.state[stable_id]
        dt = max(1, frame_idx - current["last_seen"])
        return current["uv"] + current["vel"] * min(dt, 8)

    @staticmethod
    def _side(uv: np.ndarray) -> int:
        return int(float(uv[1]) >= 0.5)

    def update(self, detections: sv.Detections, frame_idx: int) -> sv.Detections:
        if len(detections) == 0:
            for sid in list(self.state):
                if frame_idx - self.state[sid]["last_seen"] > self.max_age:
                    raw = self.state[sid].get("raw")
                    if raw in self.raw_to_stable:
                        del self.raw_to_stable[raw]
                    del self.state[sid]
            return detections

        raw_ids = (
            detections.tracker_id.astype(int)
            if detections.tracker_id is not None
            else np.arange(len(detections), dtype=int)
        )
        uvs = np.array([self._foot_uv(xyxy) for xyxy in detections.xyxy], dtype=float)
        active = [
            sid
            for sid, current in self.state.items()
            if frame_idx - current["last_seen"] <= self.max_age
        ]

        assigned: dict[int, int] = {}
        used_sids: set[int] = set()
        if active:
            pairs = []
            for row, sid in enumerate(active):
                pred = self._predict(sid, frame_idx)
                stable_side = self.state[sid].get("side")
                for col, uv in enumerate(uvs):
                    detection_side = self._side(uv)
                    if stable_side is not None and stable_side != detection_side:
                        continue
                    cost = float(np.linalg.norm(uv - pred))
                    if self.state[sid].get("raw") == int(raw_ids[col]):
                        cost -= 0.10
                    pairs.append((cost, row, col))
            for cost, row, col in sorted(pairs, key=lambda item: item[0]):
                sid = active[row]
                if cost > self.max_cost or col in assigned or sid in used_sids:
                    continue
                assigned[col] = sid
                used_sids.add(sid)

        for col, raw_id in enumerate(raw_ids):
            if col in assigned:
                continue
            detection_side = self._side(uvs[col])
            sid = self.raw_to_stable.get(int(raw_id))
            candidates = [
                (
                    float(np.linalg.norm(uvs[col] - self._predict(candidate_sid, frame_idx)))
                    - (0.10 if sid == candidate_sid else 0.0),
                    candidate_sid,
                )
                for candidate_sid in active
                if candidate_sid not in used_sids
                and self.state[candidate_sid].get("side") == detection_side
            ]
            if candidates:
                best_cost, best_sid = min(candidates, key=lambda item: item[0])
                if best_cost <= self.fallback_cost:
                    sid = best_sid
                else:
                    sid = None
            else:
                sid = None

            if sid is None:
                free_ids = [
                    candidate_sid
                    for candidate_sid in range(1, self.max_identities + 1)
                    if candidate_sid not in used_sids
                    and (
                        candidate_sid not in self.slot_sides
                        or self.slot_sides[candidate_sid] == detection_side
                    )
                ]
                if not free_ids:
                    continue
                sid = free_ids[0]
            assigned[col] = sid
            used_sids.add(sid)

        if len(assigned) < len(detections):
            keep_cols = sorted(assigned)
            detections = detections[np.array(keep_cols, dtype=int)]
            raw_ids = raw_ids[np.array(keep_cols, dtype=int)]
            uvs = uvs[np.array(keep_cols, dtype=int)]
            assigned = {new_col: assigned[old_col] for new_col, old_col in enumerate(keep_cols)}

        stable_ids = np.zeros(len(detections), dtype=int)
        for col, sid in assigned.items():
            uv = uvs[col]
            raw = int(raw_ids[col])
            old = self.state.get(sid)
            if old is None:
                vel = np.zeros(2, dtype=float)
            else:
                dt = max(1, frame_idx - old["last_seen"])
                vel = 0.35 * ((uv - old["uv"]) / dt) + 0.65 * old["vel"]
            side = self.slot_sides.get(sid, self._side(uv))
            self.slot_sides[sid] = side
            self.state[sid] = {
                "uv": uv,
                "vel": vel,
                "raw": raw,
                "side": side,
                "last_seen": frame_idx,
            }
            self.raw_to_stable[raw] = sid
            stable_ids[col] = sid

        for sid in list(self.state):
            if frame_idx - self.state[sid]["last_seen"] > self.max_age:
                raw = self.state[sid].get("raw")
                if raw in self.raw_to_stable:
                    del self.raw_to_stable[raw]
                del self.state[sid]

        detections.tracker_id = stable_ids
        return detections


def load_court(video: Path) -> np.ndarray | None:
    saved = calibration.court(video)
    if saved is not None:
        return saved

    notebook_corners = Path("notebooks/tutorials/court_corners.txt")
    if not notebook_corners.exists():
        return None
    found = {}
    for line in notebook_corners.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, x, y = line.split()
            found[name] = (float(x), float(y))
    return np.float32([found[name] for name in CORNER_NAMES])


def yolo_seg_to_detections(result, frame_shape) -> sv.Detections:
    detections = sv.Detections.from_ultralytics(result)
    if detections.class_id is not None:
        detections = detections[detections.class_id == 0]

    if result.masks is not None:
        h, w = frame_shape[:2]
        masks = []
        for poly in result.masks.xy:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
            masks.append(mask.astype(bool))
        if masks:
            detections.mask = np.array(masks, dtype=bool)
    return detections


def save_player_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def save_player_cutout(
    frame: np.ndarray,
    mask: np.ndarray,
    xyxy: np.ndarray,
    path: Path,
    padding: int = 10,
) -> tuple[str, list[int]] | tuple[None, None]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    left = max(0, int(math.floor(x1)) - padding)
    top = max(0, int(math.floor(y1)) - padding)
    right = min(w, int(math.ceil(x2)) + padding)
    bottom = min(h, int(math.ceil(y2)) + padding)
    if right <= left or bottom <= top:
        return None, None

    rgb = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2RGB)
    alpha = (mask[top:bottom, left:right].astype(np.uint8) * 255)
    rgba = np.dstack([rgb, alpha])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    return path.name, [left, top, right - left, bottom - top]


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else float(inter / union)


def matched_mask(
    xyxy: np.ndarray,
    raw_xyxy: np.ndarray,
    raw_masks: np.ndarray | None,
    used_raw: set[int],
) -> np.ndarray | None:
    if raw_masks is None or len(raw_xyxy) == 0:
        return None

    scores = [
        (bbox_iou(xyxy, candidate), idx)
        for idx, candidate in enumerate(raw_xyxy)
        if idx not in used_raw
    ]
    if not scores:
        return None

    score, idx = max(scores, key=lambda item: item[0])
    if score < 0.08:
        return None
    used_raw.add(idx)
    return raw_masks[idx]


def foot_uv_for_box(
    xyxy: np.ndarray,
    h_inv: np.ndarray | None,
    width: int,
    height: int,
) -> np.ndarray:
    x1, _y1, x2, y2 = xyxy
    foot = np.float32([[(x1 + x2) / 2, y2]]).reshape(-1, 1, 2)
    if h_inv is not None:
        return cv2.perspectiveTransform(foot, h_inv).reshape(-1)
    return np.array([foot[0, 0, 0] / width, foot[0, 0, 1] / height], dtype=float)


def keep_main_players(
    detections: sv.Detections,
    h_inv: np.ndarray | None,
    width: int,
    height: int,
    max_players: int = 4,
) -> sv.Detections:
    """Keep the four most plausible on-court player detections.

    This prevents spectators/reflections/old coasted tracks from entering the
    stable-ID mapper, which is what causes P1..P40 style identity churn.
    """
    if len(detections) <= max_players:
        return detections

    confs = detections.confidence if detections.confidence is not None else np.ones(len(detections))
    scored = []
    for idx, xyxy in enumerate(detections.xyxy):
        uv = foot_uv_for_box(xyxy, h_inv, width, height)
        inside = 0 <= uv[0] <= 1 and 0 <= uv[1] <= 1
        x1, y1, x2, y2 = xyxy
        area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
        center_bias = -0.15 * float(abs(uv[0] - 0.5))
        score = (
            (3.0 if inside else -3.0)
            + float(confs[idx])
            + min(area / 25000.0, 1.0)
            + center_bias
        )
        scored.append((score, idx))

    keep = sorted(idx for _score, idx in sorted(scored, reverse=True)[:max_players])
    return detections[np.array(keep, dtype=int)]


def make_heatmap_snapshot(grids: dict[int, np.ndarray], nx: int, ny: int) -> dict:
    if grids:
        total = np.sum(list(grids.values()), axis=0)
    else:
        total = np.zeros((ny, nx), np.float32)
    total = total / max(float(total.max()), 1.0)
    return {"nx": nx, "ny": ny, "values": total.round(4).tolist()}


def speed_for_track(
    history: dict[int, list[tuple[int, float, float]]],
    tid: int,
    fps: float,
) -> float:
    points = history.get(tid, [])
    if len(points) < 2:
        return 0.0
    f0, x0, y0 = points[-2]
    f1, x1, y1 = points[-1]
    dt = max((f1 - f0) / fps, 1e-6)
    px_per_second = math.hypot(x1 - x0, y1 - y0) / dt
    return px_per_second / 18.0


def move_for_track(
    history: dict[int, list[tuple[int, float, float]]],
    tid: int,
) -> list[float] | None:
    points = history.get(tid, [])
    if len(points) < 2:
        return None
    _f0, x0, y0 = points[-2]
    _f1, x1, y1 = points[-1]
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1e-6:
        return None
    return [round((x1 - x0) / length, 4), round((y1 - y0) / length, 4)]


def state_for_player(uv: np.ndarray, speed: float) -> str:
    if speed > 14:
        return "SPRINT"
    if uv[1] < 0.42:
        return "NET"
    if speed > 7:
        return "COVER"
    return "TRACKING"


def export(args: argparse.Namespace) -> Path:
    from ultralytics import YOLO

    video = Path(args.video)
    out_dir = Path(args.output)
    frames_dir = out_dir / "frames"
    masks_dir = out_dir / "masks"
    cutouts_dir = out_dir / "cutouts"

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    cutouts_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(args.start * fps)
    end_frame = (
        total_frames
        if args.seconds is None
        else min(total_frames, start_frame + int(args.seconds * fps))
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    court = load_court(video)
    h_inv = cv2.getPerspectiveTransform(court, UNIT) if court is not None else None
    roi = calibration.roi(video)
    zone = sv.PolygonZone(polygon=roi) if roi is not None else None

    model = YOLO(args.model)
    tracker = sv.ByteTrack(frame_rate=int(round(fps)), lost_track_buffer=int(round(fps)))
    identity = (
        StableIdentityMapper(h_inv, max_identities=args.max_players)
        if h_inv is not None
        else None
    )
    stabilizer = TrackStabilizer(hold_frames=int(round(fps)))
    grids = defaultdict(lambda: np.zeros((args.ny, args.nx), np.float32))
    history: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    records = []

    for frame_idx in tqdm(range(start_frame, end_frame), desc="export remotion data"):
        ok, frame = cap.read()
        if not ok:
            break
        if (frame_idx - start_frame) % args.stride:
            continue

        frame_name = f"frame_{frame_idx:06d}.jpg"
        frame_path = frames_dir / frame_name
        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)])

        result = model.predict(
            frame,
            conf=args.conf,
            classes=[0],
            retina_masks=True,
            verbose=False,
        )[0]
        detections = yolo_seg_to_detections(result, frame.shape)
        if zone is not None:
            detections = detections[zone.trigger(detections)]
        raw_xyxy = detections.xyxy.copy()
        raw_masks = detections.mask.copy() if detections.mask is not None else None
        detections = tracker.update_with_detections(detections)
        detections = keep_main_players(
            detections,
            h_inv,
            width,
            height,
            max_players=args.max_players,
        )
        if identity is not None:
            detections = identity.update(detections, frame_idx)
        detections = stabilizer.update(detections, frame_idx)
        detections = keep_main_players(
            detections,
            h_inv,
            width,
            height,
            max_players=args.max_players,
        )

        players = []
        used_raw_masks: set[int] = set()
        for det_idx, xyxy in enumerate(detections.xyxy):
            tracker_id = (
                int(detections.tracker_id[det_idx])
                if detections.tracker_id is not None
                else det_idx + 1
            )
            x1, y1, x2, y2 = [float(x) for x in xyxy]
            foot = np.array([(x1 + x2) / 2, y2], dtype=float)
            uv = (
                cv2.perspectiveTransform(
                    foot.reshape(-1, 1, 2).astype(np.float32),
                    h_inv,
                ).reshape(-1)
                if h_inv is not None
                else np.array([foot[0] / width, foot[1] / height])
            )
            if 0 <= uv[0] < 1 and 0 <= uv[1] < 1:
                grids[tracker_id][
                    int(np.clip(uv[1] * args.ny, 0, args.ny - 1)),
                    int(np.clip(uv[0] * args.nx, 0, args.nx - 1)),
                ] += 1

            history[tracker_id].append((frame_idx, float(foot[0]), float(foot[1])))
            history[tracker_id] = history[tracker_id][-8:]
            speed = speed_for_track(history, tracker_id, fps)
            move = move_for_track(history, tracker_id)
            mask_rel = None
            cutout_rel = None
            cutout_box = None
            mask = matched_mask(xyxy, raw_xyxy, raw_masks, used_raw_masks)
            if mask is not None:
                mask_name = f"frame_{frame_idx:06d}_p{tracker_id}.png"
                save_player_mask(mask, masks_dir / mask_name)
                mask_rel = f"render/masks/{mask_name}"
                cutout_name = f"frame_{frame_idx:06d}_p{tracker_id}.png"
                saved_name, cutout_box = save_player_cutout(
                    frame,
                    mask,
                    xyxy,
                    cutouts_dir / cutout_name,
                    padding=args.cutout_padding,
                )
                if saved_name is not None:
                    cutout_rel = f"render/cutouts/{saved_name}"

            players.append(
                {
                    "id": f"P{tracker_id}",
                    "team": "A" if tracker_id in (1, 2) else "B",
                    "color": PALETTE[(tracker_id - 1) % len(PALETTE)],
                    "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "cutout": cutout_rel,
                    "cutoutBox": cutout_box,
                    "feet": [round(float(foot[0]), 2), round(float(foot[1]), 2)],
                    "head": [round((x1 + x2) / 2, 2), round(y1, 2)],
                    "speed": round(float(speed), 2),
                    "state": state_for_player(uv, speed),
                    "move": move,
                    "mask": mask_rel,
                }
            )

        records.append(
            {
                "frame": len(records),
                "sourceFrame": frame_idx,
                "time": round(frame_idx / fps, 4),
                "sourceWidth": width,
                "sourceHeight": height,
                "image": f"render/frames/{frame_name}",
                "players": players,
                "heatmap": make_heatmap_snapshot(grids, args.nx, args.ny),
            }
        )

    cap.release()

    payload = {
        "fps": fps / max(1, args.stride),
        "sourceFps": fps,
        "sourceWidth": width,
        "sourceHeight": height,
        "frames": records,
    }
    output = out_dir / "frames.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Exported {len(records)} frames -> {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Path to source padel video")
    parser.add_argument(
        "--output",
        default="renderer/public/render",
        help="Remotion public output dir",
    )
    parser.add_argument("--model", default="yolo11n-seg.pt", help="Segmentation model weights")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Duration to export in seconds; omit to export until the end",
    )
    parser.add_argument("--clean", action="store_true", help="Delete old exported frames first")
    parser.add_argument("--stride", type=int, default=2, help="Export every Nth source frame")
    parser.add_argument("--conf", type=float, default=0.45, help="Detection confidence")
    parser.add_argument("--max-players", type=int, default=4, help="Maximum tracked players")
    parser.add_argument("--nx", type=int, default=12, help="Heatmap columns")
    parser.add_argument("--ny", type=int, default=8, help="Heatmap rows")
    parser.add_argument("--jpeg-quality", type=int, default=92, help="Exported frame JPEG quality")
    parser.add_argument(
        "--cutout-padding",
        type=int,
        default=10,
        help="Player cutout padding in px",
    )
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
