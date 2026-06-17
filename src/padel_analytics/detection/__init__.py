"""Detection: turn video frames into canonical player/ball detections + overlays."""

from .annotate import DetectionAnnotator
from .classes import BALL, CANONICAL_NAMES, PLAYER
from .detector import ObjectDetector, RFDETRDetector, YOLODetector, build_detector

__all__ = [
    "ObjectDetector",
    "RFDETRDetector",
    "YOLODetector",
    "build_detector",
    "DetectionAnnotator",
    "PLAYER",
    "BALL",
    "CANONICAL_NAMES",
]
