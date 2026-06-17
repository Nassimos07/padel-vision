"""Lightweight tests that don't require torch/rfdetr/ultralytics to be installed."""

from __future__ import annotations

from textwrap import dedent

from padel_vision import __version__
from padel_vision.config import Config


def test_version():
    assert __version__


def test_default_config():
    cfg = Config()
    assert cfg.detector.backend == "rfdetr"
    assert cfg.detector.model == "medium"
    assert 0.0 < cfg.detector.confidence < 1.0
    assert cfg.annotation.show_labels is True


def test_config_from_yaml(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(dedent("""
            detector:
              backend: yolo
              model: yolo11s.pt
              confidence: 0.4
            annotation:
              show_confidence: false
            """))
    cfg = Config.from_yaml(path)
    assert cfg.detector.backend == "yolo"
    assert cfg.detector.model == "yolo11s.pt"
    assert cfg.detector.confidence == 0.4
    assert cfg.annotation.show_confidence is False
    # Untouched fields keep their defaults.
    assert cfg.detector.iou == 0.50


def test_canonical_classes():
    from padel_vision.detection.classes import (
        BALL,
        PLAYER,
        RFDETR_TO_CANONICAL,
        YOLO_TO_CANONICAL,
    )

    assert PLAYER != BALL
    assert YOLO_TO_CANONICAL[0] == PLAYER  # COCO-80 person
    assert RFDETR_TO_CANONICAL[1] == PLAYER  # COCO-91 person


def test_court_corner_roundtrip(tmp_path):
    from padel_vision.court import CORNER_NAMES, load_corners, save_corners

    corners = [(715, 240), (1250, 240), (1530, 1070), (455, 1065)]
    path = save_corners(corners, tmp_path / "court_corners.txt")
    loaded = load_corners(path)
    assert list(CORNER_NAMES) == ["TL", "TR", "BR", "BL"]
    assert loaded.shape == (4, 2)
    assert loaded.tolist() == [list(map(float, c)) for c in corners]
