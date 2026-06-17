"""Small wrappers around video reading/writing and browser-friendly transcoding."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import supervision as sv


def video_info(source: str | Path) -> sv.VideoInfo:
    return sv.VideoInfo.from_video_path(str(source))


def frame_generator(source: str | Path, stride: int = 1) -> Iterator[np.ndarray]:
    return sv.get_video_frames_generator(str(source), stride=stride)


def grab_frame(source: str | Path, index: int = 0) -> np.ndarray:
    """Random-access a single BGR frame by index."""
    cap = cv2.VideoCapture(str(source))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise FileNotFoundError(f"could not read frame {index} from {source!r}")
    return frame


def to_h264(source: str | Path, target: str | Path | None = None) -> Path:
    """Transcode a video to H.264/AAC so browsers (and ``st.video``) can play it.

    Falls back to the original file if ffmpeg is unavailable.
    """
    source = Path(source)
    if shutil.which("ffmpeg") is None:
        return source
    target = Path(target) if target else source.with_name(f"{source.stem}_h264.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(target),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return target
