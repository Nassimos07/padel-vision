"""``padel-vision`` command-line interface, built with `Python Fire`_.

Examples::

    padel-vision court adjust data/raw/match.mp4
    padel-vision court adjust data/raw/match.mp4 --frame 500 --output court_corners.txt

.. _Python Fire: https://github.com/google/python-fire
"""

from __future__ import annotations

import fire

from padel_vision import __version__
from padel_vision.court import adjust as _court_adjust


class _Court:
    """Court geometry tools."""

    def adjust(self, video: str, output: str = "court_corners.txt", frame: int = 300):
        """Pick the 4 court corners (TL, TR, BR, BL) on a frame and save them.

        Args:
            video: path to a padel clip.
            output: where to write the corners (default ``court_corners.txt``).
            frame: frame index to display for picking (default ``300``).
        """
        return _court_adjust(video, output=output, frame=frame)


class PadelML:
    """padel-vision — computer-vision analytics for padel."""

    def __init__(self):
        self.court = _Court()

    def version(self):
        """Print the installed version."""
        return __version__


def main():
    fire.Fire(PadelML, name="padel-vision")


if __name__ == "__main__":
    main()
