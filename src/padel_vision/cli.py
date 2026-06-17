"""``padel-vision`` command-line interface, built with `Python Fire`_.

A small calibrate → detect workflow, all keyed automatically by the clip's name:

    padel-vision roi adjust     data/raw/match.mp4   # draw a detection-filter polygon
    padel-vision court adjust   data/raw/match.mp4   # pick the 4 court corners
    padel-vision detect players data/raw/match.mp4   # detect, filtered by the saved ROI

Calibration is stored per clip under ``data/calibration/<stem>.json``.

.. _Python Fire: https://github.com/google/python-fire
"""

from __future__ import annotations

import fire

from padel_vision import __version__, calibration
from padel_vision.court import CORNER_NAMES
from padel_vision.detect import detect_players
from padel_vision.heatmap import make_heatmap
from padel_vision.pickers import pick_points
from padel_vision.video.io import grab_frame


class _Roi:
    """Region-of-interest tools — the polygon that filters detections."""

    def adjust(self, video: str, frame: int = 300):
        """Draw a polygon ROI on a frame and save it for this clip.

        Args:
            video: path to a padel clip.
            frame: frame index to display for picking (default ``300``).
        """
        pts = pick_points(
            grab_frame(video, frame), n=None,
            title="ROI - click the polygon outline, then 'f' to finish",
        )
        if pts is None:
            print("cancelled — nothing saved")
            return
        print(f"saved ROI ({len(pts)} points) -> {calibration.save_roi(video, pts)}")


class _Court:
    """Court geometry — the 4 corners used for the heatmap homography."""

    def adjust(self, video: str, frame: int = 300):
        """Pick the 4 court corners (TL, TR, BR, BL) and save them for this clip."""
        pts = pick_points(
            grab_frame(video, frame), n=4, labels=CORNER_NAMES,
            title="Court - click TL, TR, BR, BL",
        )
        if pts is None:
            print("cancelled — nothing saved")
            return
        print(f"saved court corners -> {calibration.save_court(video, pts)}")


class _Detect:
    """Detection commands."""

    def players(self, video: str, frame: int = 0, conf: float = 0.5,
                model: str = "medium", stride: int = 2):
        """Detect players in real time, filtered by the saved ROI if there is one.

        Args:
            video: path to a padel clip.
            frame: frame index to start from (default ``0``).
            conf: detection confidence threshold (default ``0.5``).
            model: RF-DETR size — nano|small|medium|base|large (smaller = faster).
            stride: detect every Nth frame, reusing the result in between (smoother stream).
        """
        detect_players(video, frame=frame, conf=conf, model=model, stride=stride)


class PadelVision:
    """padel-vision — computer-vision analytics for padel."""

    def __init__(self):
        self.roi = _Roi()
        self.court = _Court()
        self.detect = _Detect()

    def heatmap(self, video: str, start: float = 0.0, duration: float = None,
                stride: int = 3, conf: float = 0.5, model: str = "medium",
                output: str = None, show: bool = True):
        """Render the zonal court heatmap for a clip (run `court adjust` first).

        Args:
            video: path to a padel clip.
            start: seconds to start from (default 0).
            duration: seconds to accumulate (default: the whole clip).
            stride: process every Nth frame (default 3; higher = faster).
            conf: detection confidence threshold (default 0.5).
            model: RF-DETR size — nano|small|medium|base|large (smaller = faster).
            output: image path (default data/processed/<clip>_heatmap.jpg).
            show: also display the result in a window (default True).
        """
        make_heatmap(video, start=start, duration=duration, stride=stride,
                     conf=conf, model=model, output=output, show=show)

    def version(self):
        """Print the installed version."""
        return __version__


def main():
    fire.Fire(PadelVision, name="padel-vision")


if __name__ == "__main__":
    main()
