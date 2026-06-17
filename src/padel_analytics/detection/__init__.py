"""Detection: turn video frames into player/ball detections and pretty overlays."""

from .annotate import DetectionAnnotator
from .detector import ObjectDetector, YOLODetector

__all__ = ["ObjectDetector", "YOLODetector", "DetectionAnnotator"]
