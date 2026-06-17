"""Object detectors.

A thin, swappable interface sits in front of each concrete model so we can mix
and match backends (RF-DETR for players now, a custom ball model later) without
touching the rest of the package. Every detector returns ``sv.Detections`` in
the canonical class space (see ``classes.py``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np
import supervision as sv

from ..config import DetectorConfig
from .classes import (
    BALL,
    RFDETR_TO_CANONICAL,
    YOLO_TO_CANONICAL,
    remap_to_canonical,
)


class ObjectDetector(ABC):
    """Detect objects in a single BGR frame and return canonical ``sv.Detections``."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> sv.Detections:
        raise NotImplementedError


class RFDETRDetector(ObjectDetector):
    """RF-DETR (Roboflow) detector — COCO-pretrained, transformer-based, no NMS."""

    _CLASS_TABLE = {
        "nano": "RFDETRNano",
        "small": "RFDETRSmall",
        "medium": "RFDETRMedium",
        "base": "RFDETRBase",
        "large": "RFDETRLarge",
    }

    def __init__(self, config: DetectorConfig | None = None) -> None:
        import rfdetr
        import torch

        self.config = config or DetectorConfig(backend="rfdetr")
        # RF-DETR defaults to "cuda"; resolve our auto semantics (None => cuda if
        # available, else cpu) so CPU-only machines don't fail on the default.
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_cls = self._resolve_model_class(rfdetr, self.config.model)
        self.model = model_cls(device=self.device)

        # Reuse the single source of truth for RF-DETR's COCO ids; drop the ball
        # class unless explicitly requested (mirrors how YOLODetector filters).
        self.mapping = {
            coco_id: canonical
            for coco_id, canonical in RFDETR_TO_CANONICAL.items()
            if canonical != BALL or self.config.detect_ball
        }

    @classmethod
    def _resolve_model_class(cls, rfdetr_module, name: str):
        class_name = cls._CLASS_TABLE.get(name.lower(), "RFDETRBase")
        if not hasattr(rfdetr_module, class_name):  # older versions only ship Base/Large
            class_name = "RFDETRBase"
        return getattr(rfdetr_module, class_name)

    def detect(self, frame: np.ndarray) -> sv.Detections:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self.model.predict(rgb, threshold=self.config.confidence)
        return remap_to_canonical(detections, self.mapping)


class YOLODetector(ObjectDetector):
    """Ultralytics YOLO (v11) detector — kept as a fast, swappable alternative."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        import torch
        from ultralytics import YOLO

        self.config = config or DetectorConfig(backend="yolo", model="yolo11m.pt")
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(self.config.model)
        self.model.to(self.device)

        self._classes = [0] + ([32] if self.config.detect_ball else [])

    def detect(self, frame: np.ndarray) -> sv.Detections:
        result = self.model.predict(
            frame,
            conf=self.config.confidence,
            iou=self.config.iou,
            classes=self._classes,
            imgsz=self.config.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        return remap_to_canonical(detections, YOLO_TO_CANONICAL)


def build_detector(config: DetectorConfig) -> ObjectDetector:
    """Instantiate the detector backend named in the config."""
    backend = config.backend.lower()
    if backend == "rfdetr":
        return RFDETRDetector(config)
    if backend == "yolo":
        return YOLODetector(config)
    raise ValueError(f"Unknown detector backend: {config.backend!r} (use 'rfdetr' or 'yolo')")
