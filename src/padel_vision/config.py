"""Typed, YAML-backed detector configuration for padel-vision."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DetectorConfig:
    backend: str = "rfdetr"  # "rfdetr" | "yolo"
    # rfdetr: one of {nano, small, medium, base, large}
    # yolo:   a weights name/path, e.g. "yolo11m.pt"
    model: str = "medium"
    confidence: float = 0.5
    iou: float = 0.50  # YOLO NMS threshold (ignored by RF-DETR)
    imgsz: int = 640  # YOLO inference size (ignored by RF-DETR)
    device: str | None = None  # None => auto (cuda if available, else cpu)
    detect_ball: bool = False  # ball is tiny/fast — off by default (dedicated stage later)


@dataclass
class AnnotationConfig:
    show_labels: bool = True
    show_confidence: bool = True


@dataclass
class Config:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a config from YAML, falling back to defaults for missing keys."""
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            detector=DetectorConfig(**(data.get("detector") or {})),
            annotation=AnnotationConfig(**(data.get("annotation") or {})),
        )
