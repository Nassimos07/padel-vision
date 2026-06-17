"""Tracking helpers that do not require GUI windows or model downloads."""

from __future__ import annotations

import numpy as np
import supervision as sv

from padel_vision.track import TrackStabilizer


def _detection(x1: float, tracker_id: int = 7) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([[x1, 10, x1 + 20, 40]], dtype=float),
        confidence=np.array([0.9], dtype=float),
        class_id=np.zeros(1, dtype=int),
        tracker_id=np.array([tracker_id], dtype=int),
    )


def test_track_stabilizer_smooths_and_coasts():
    stabilizer = TrackStabilizer(smoothing=0.5, hold_frames=2)

    first = stabilizer.update(_detection(10), frame_idx=0)
    assert first.tracker_id.tolist() == [7]
    assert first.xyxy.tolist() == [[10.0, 10.0, 30.0, 40.0]]

    second = stabilizer.update(_detection(30), frame_idx=1)
    assert second.xyxy.tolist() == [[20.0, 10.0, 40.0, 40.0]]

    coast = stabilizer.update(sv.Detections.empty(), frame_idx=3)
    assert coast.tracker_id.tolist() == [7]
    assert coast.xyxy.tolist() == [[20.0, 10.0, 40.0, 40.0]]

    expired = stabilizer.update(sv.Detections.empty(), frame_idx=4)
    assert len(expired) == 0
