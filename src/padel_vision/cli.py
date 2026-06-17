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
from padel_vision.rings import adjust_ring
from padel_vision.track import track_players
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

    def ring(self, video: str, frame: int = 300, model: str = "medium"):
        """Tune the AR ground-ring look (radius, tilt, perspective, glow, ...) and save it.

        Opens an interactive window with sliders (like the notebook's "Dial it in" cell),
        previewed on the players in the frame; press 's' to save, 'q' to cancel.

        Args:
            video: path to a padel clip.
            frame: frame index to preview on (default 300).
            model: RF-DETR size used for the preview detection (default medium).
        """
        adjust_ring(video, frame=frame, model=model)


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


class _Track:
    """Tracking commands."""

    def players(
        self,
        video: str,
        frame: int = 0,
        conf: float = 0.5,
        model: str = "medium",
        stride: int = 1,
        smoothing: float = 0.35,
        hold_frames: int = 30,
        foreground: bool = True,
        foreground_model: str = "yolo11n-seg.pt",
        foreground_stride: int = 1,
        trail: bool = False,
        labels: bool = False,
    ):
        """Track players live with the full AR overlay from the v1 notebook.

        Args:
            video: path to a padel clip.
            frame: frame index to start from (default ``0``).
            conf: detection confidence threshold (default ``0.5``).
            model: RF-DETR size — nano|small|medium|base|large (smaller = faster).
            stride: run detection/tracking every Nth frame and coast between.
            smoothing: EMA weight for box smoothing (lower = steadier, more lag).
            hold_frames: frames to keep a lost track alive.
            foreground: segment and paste players in front of the AR graphics.
            foreground_model: Ultralytics segmentation weights for foreground matting.
            foreground_stride: refresh the foreground matte every Nth frame.
            trail: show a short movement trail behind each tracked player.
            labels: show P<ID> labels under players.
        """
        track_players(
            video,
            frame=frame,
            conf=conf,
            model=model,
            stride=stride,
            smoothing=smoothing,
            hold_frames=hold_frames,
            foreground=foreground,
            foreground_model=foreground_model,
            foreground_stride=foreground_stride,
            trail=trail,
            labels=labels,
        )


class PadelVision:
    """padel-vision — computer-vision analytics for padel."""

    def __init__(self):
        self.roi = _Roi()
        self.court = _Court()
        self.detect = _Detect()
        self.track = _Track()

    def heatmap(self, video: str, start: float = 0.0, duration: float = None,
                stride: int = 3, conf: float = 0.5, model: str = "medium",
                output: str = None, show: bool = True, frame: int = None,
                foreground: bool = True, foreground_model: str = "yolo11n-seg.pt",
                trail: bool = False, labels: bool = False):
        """Render one heatmap preview frame with segmentation + detections.

        Args:
            video: path to a padel clip.
            start: seconds to pick the preview frame from (default 0).
            duration: accepted for compatibility; ignored in single-frame preview mode.
            stride: accepted for compatibility; ignored in single-frame preview mode.
            conf: detection confidence threshold (default 0.5).
            model: RF-DETR size — nano|small|medium|base|large (smaller = faster).
            output: image path (default data/processed/<clip>_heatmap.jpg).
            show: also display the result in a window (default True).
            frame: exact frame index to preview; overrides ``start`` when set.
            foreground: segment and paste players in front of the heatmap.
            foreground_model: Ultralytics segmentation weights for foreground matting.
            trail: show the same movement trail annotation used by ``track players``.
            labels: show the same P<ID> labels used by ``track players``.
        """
        make_heatmap(video, start=start, duration=duration, stride=stride,
                     conf=conf, model=model, output=output, show=show,
                     frame=frame, foreground=foreground, foreground_model=foreground_model,
                     trail=trail, labels=labels)

    def version(self):
        """Print the installed version."""
        return __version__


def main():
    fire.Fire(PadelVision, name="padel-vision")


if __name__ == "__main__":
    main()
