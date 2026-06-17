"""Lightweight tests that don't require torch/rfdetr/ultralytics to be installed."""

from __future__ import annotations

from textwrap import dedent

from padel_analytics import __version__
from padel_analytics.config import Config


def test_version():
    assert __version__


def test_default_config():
    cfg = Config()
    assert cfg.detector.backend == "rfdetr"
    assert cfg.detector.model == "medium"
    assert 0.0 < cfg.detector.confidence < 1.0
    assert cfg.video.stride >= 1


def test_config_from_yaml(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(dedent("""
            detector:
              backend: yolo
              model: yolo11s.pt
              confidence: 0.4
            video:
              stride: 2
            """))
    cfg = Config.from_yaml(path)
    assert cfg.detector.backend == "yolo"
    assert cfg.detector.model == "yolo11s.pt"
    assert cfg.detector.confidence == 0.4
    assert cfg.video.stride == 2
    # Untouched fields keep their defaults.
    assert cfg.detector.iou == 0.50


def test_canonical_classes():
    from padel_analytics.detection.classes import (
        BALL,
        PLAYER,
        RFDETR_TO_CANONICAL,
        YOLO_TO_CANONICAL,
    )

    assert PLAYER != BALL
    assert YOLO_TO_CANONICAL[0] == PLAYER  # COCO-80 person
    assert RFDETR_TO_CANONICAL[1] == PLAYER  # COCO-91 person


def test_detection_stats():
    from padel_analytics.pipeline import DetectionStats

    stats = DetectionStats(frames=10, total_players=38, ball_frames=6)
    assert stats.avg_players_per_frame == 3.8
    assert stats.ball_visible_pct == 60.0
