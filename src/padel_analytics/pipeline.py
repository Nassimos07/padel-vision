"""End-to-end pipeline runners. Stage 1: detection overlay + basic frame stats."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import supervision as sv
from tqdm import tqdm

from .config import BALL_CLASS_ID, PERSON_CLASS_ID, Config
from .detection.annotate import DetectionAnnotator
from .detection.detector import ObjectDetector, YOLODetector


@dataclass
class DetectionStats:
    frames: int = 0
    total_players: int = 0
    ball_frames: int = 0

    @property
    def avg_players_per_frame(self) -> float:
        return self.total_players / self.frames if self.frames else 0.0

    @property
    def ball_visible_pct(self) -> float:
        return 100 * self.ball_frames / self.frames if self.frames else 0.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["avg_players_per_frame"] = round(self.avg_players_per_frame, 2)
        d["ball_visible_pct"] = round(self.ball_visible_pct, 1)
        return d


def run_detection(
    source: str | Path,
    target: str | Path,
    config: Config | None = None,
    detector: ObjectDetector | None = None,
    progress: bool = True,
) -> DetectionStats:
    """Detect players + ball on every frame and write an annotated video.

    Returns simple per-video statistics.
    """
    config = config or Config()
    detector = detector or YOLODetector(config.detector)
    annotator = DetectionAnnotator(config.annotation)

    info = sv.VideoInfo.from_video_path(str(source))
    stride = max(config.video.stride, 1)
    total = info.total_frames // stride if info.total_frames else None
    frames = sv.get_video_frames_generator(str(source), stride=stride)

    Path(target).parent.mkdir(parents=True, exist_ok=True)
    stats = DetectionStats()

    with sv.VideoSink(str(target), video_info=info) as sink:
        for frame in tqdm(frames, total=total, desc="Detecting", disable=not progress):
            detections = detector.detect(frame)
            sink.write_frame(annotator.annotate(frame, detections))

            stats.frames += 1
            class_id = detections.class_id
            if class_id is not None and len(class_id):
                stats.total_players += int((class_id == PERSON_CLASS_ID).sum())
                if bool((class_id == BALL_CLASS_ID).any()):
                    stats.ball_frames += 1

    return stats
