"""Object detectors.

A thin, swappable interface sits in front of the concrete model so that we can
move from YOLO to RF-DETR (or anything else) later without touching the rest of
the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import supervision as sv

from ..config import DetectorConfig


class ObjectDetector(ABC):
    """Detect objects in a single BGR frame and return ``sv.Detections``."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> sv.Detections:
        raise NotImplementedError


class YOLODetector(ObjectDetector):
    """Ultralytics YOLO (v11) detector, filtered to the configured classes."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        # Imported lazily so the package can be imported without torch installed.
        import torch
        from ultralytics import YOLO

        self.config = config or DetectorConfig()
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(self.config.model_path)
        self.model.to(self.device)

    @property
    def class_names(self) -> dict[int, str]:
        return self.model.names

    def detect(self, frame: np.ndarray) -> sv.Detections:
        result = self.model.predict(
            frame,
            conf=self.config.confidence,
            iou=self.config.iou,
            classes=self.config.classes,
            imgsz=self.config.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(result)
