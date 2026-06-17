"""Calibration store + ROI filtering (no GUI, no model downloads)."""

from __future__ import annotations

import numpy as np

from padel_vision import calibration


def test_roi_court_roundtrip(tmp_path):
    video = "match.mp4"
    assert calibration.load(video, tmp_path) == {}

    calibration.save_roi(video, [(10, 20), (30, 20), (30, 40), (10, 40)], base=tmp_path)
    calibration.save_court(video, [(0, 0), (100, 0), (100, 50), (0, 50)], base=tmp_path)

    data = calibration.load(video, tmp_path)
    assert data["video"] == "match.mp4"
    assert data["roi"] == [[10, 20], [30, 20], [30, 40], [10, 40]]
    assert list(data["court"]) == ["TL", "TR", "BR", "BL"]

    assert calibration.roi(video, tmp_path).tolist() == [[10, 20], [30, 20], [30, 40], [10, 40]]
    assert calibration.court(video, tmp_path).shape == (4, 2)
    assert calibration.path_for(video, tmp_path).name == "match.json"


def test_missing_calibration(tmp_path):
    assert calibration.roi("nope.mp4", tmp_path) is None
    assert calibration.court("nope.mp4", tmp_path) is None


def test_roi_filters_detections():
    import supervision as sv

    roi = np.array([(100, 100), (300, 100), (300, 300), (100, 300)], np.int32)
    zone = sv.PolygonZone(polygon=roi)
    dets = sv.Detections(
        xyxy=np.array([[150, 150, 250, 250],   # inside the ROI
                       [400, 400, 450, 450]],  # outside
                      dtype=float),
        class_id=np.zeros(2, dtype=int),
    )
    assert zone.trigger(dets).tolist() == [True, False]
