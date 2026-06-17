"""Typed, YAML-backed configuration for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# COCO class ids used by the default YOLO models.
PERSON_CLASS_ID = 0  # padel players
BALL_CLASS_ID = 32  # "sports ball"


@dataclass
class DetectorConfig:
    model_path: str = "yolo11m.pt"
    confidence: float = 0.30
    iou: float = 0.50
    imgsz: int = 640
    device: str | None = None  # None => auto (cuda if available, else cpu)
    classes: list[int] = field(default_factory=lambda: [PERSON_CLASS_ID, BALL_CLASS_ID])


@dataclass
class VideoConfig:
    stride: int = 1  # process every Nth frame


@dataclass
class AnnotationConfig:
    show_labels: bool = True
    show_confidence: bool = True


@dataclass
class Config:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load a config from a YAML file, falling back to defaults for missing keys."""
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            detector=DetectorConfig(**(data.get("detector") or {})),
            video=VideoConfig(**(data.get("video") or {})),
            annotation=AnnotationConfig(**(data.get("annotation") or {})),
        )
