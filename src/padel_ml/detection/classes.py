"""Canonical class space shared across detector backends.

Detectors disagree on COCO indexing: Ultralytics YOLO uses the 80-class,
0-indexed space (``person=0``, ``sports ball=32``), while RF-DETR uses the
original COCO 91-category space (``person=1``, ``sports ball=37``). We normalize
everything into a tiny canonical set so the rest of the pipeline never has to
care which model produced a detection.
"""

from __future__ import annotations

import numpy as np
import supervision as sv

# --- Canonical classes -----------------------------------------------------
PLAYER = 0
BALL = 1
CANONICAL_NAMES: dict[int, str] = {PLAYER: "player", BALL: "ball"}

# --- Native COCO id -> canonical id ----------------------------------------
YOLO_TO_CANONICAL: dict[int, int] = {0: PLAYER, 32: BALL}  # Ultralytics COCO-80
RFDETR_TO_CANONICAL: dict[int, int] = {1: PLAYER, 37: BALL}  # COCO-91 category ids


def remap_to_canonical(detections: sv.Detections, mapping: dict[int, int]) -> sv.Detections:
    """Keep only the mapped classes and relabel them into the canonical space."""
    if detections.class_id is None or len(detections) == 0:
        return detections

    keep = np.array([int(c) in mapping for c in detections.class_id], dtype=bool)
    detections = detections[keep]
    if len(detections) == 0:
        return detections

    canonical = np.array([mapping[int(c)] for c in detections.class_id], dtype=int)
    detections.class_id = canonical
    detections.data["class_name"] = np.array([CANONICAL_NAMES[int(c)] for c in canonical])
    return detections
