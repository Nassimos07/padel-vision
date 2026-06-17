"""Lightweight tests that don't require torch/ultralytics to be installed."""

from __future__ import annotations

from textwrap import dedent

from padel_analytics import __version__
from padel_analytics.config import BALL_CLASS_ID, PERSON_CLASS_ID, Config


def test_version():
    assert __version__


def test_default_config():
    cfg = Config()
    assert cfg.detector.classes == [PERSON_CLASS_ID, BALL_CLASS_ID]
    assert 0.0 < cfg.detector.confidence < 1.0
    assert cfg.video.stride >= 1


def test_config_from_yaml(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        dedent(
            """
            detector:
              model_path: yolo11s.pt
              confidence: 0.5
            video:
              stride: 2
            """
        )
    )
    cfg = Config.from_yaml(path)
    assert cfg.detector.model_path == "yolo11s.pt"
    assert cfg.detector.confidence == 0.5
    assert cfg.video.stride == 2
    # Untouched fields keep their defaults.
    assert cfg.detector.iou == 0.50


def test_detection_stats():
    from padel_analytics.pipeline import DetectionStats

    stats = DetectionStats(frames=10, total_players=38, ball_frames=6)
    assert stats.avg_players_per_frame == 3.8
    assert stats.ball_visible_pct == 60.0
