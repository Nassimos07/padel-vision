"""Draw detections onto frames with Supervision annotators."""

from __future__ import annotations

import numpy as np
import supervision as sv

from ..config import AnnotationConfig


class DetectionAnnotator:
    """Render bounding boxes + labels for player/ball detections."""

    def __init__(self, config: AnnotationConfig | None = None) -> None:
        self.config = config or AnnotationConfig()
        self.box_annotator = sv.RoundBoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=4)

    def _labels(self, detections: sv.Detections) -> list[str]:
        names = detections.data.get("class_name", [])
        labels: list[str] = []
        for i in range(len(detections)):
            name = names[i] if i < len(names) else str(detections.class_id[i])
            if self.config.show_confidence and detections.confidence is not None:
                labels.append(f"{name} {detections.confidence[i]:.2f}")
            else:
                labels.append(str(name))
        return labels

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        out = self.box_annotator.annotate(frame.copy(), detections)
        if self.config.show_labels and len(detections):
            out = self.label_annotator.annotate(out, detections, self._labels(detections))
        return out
