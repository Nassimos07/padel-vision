"""Command-line interface: ``padel detect <video>``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config
from .pipeline import run_detection


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="padel", description="Padel match analytics")
    sub = parser.add_subparsers(dest="command", required=True)

    detect = sub.add_parser("detect", help="Run object detection on a video")
    detect.add_argument("source", type=str, help="Input video path")
    detect.add_argument("-o", "--output", type=str, default=None, help="Output video path")
    detect.add_argument("-c", "--config", type=str, default=None, help="YAML config path")
    detect.add_argument("--model", type=str, default=None, help="Override model (e.g. yolo11s.pt)")
    detect.add_argument("--conf", type=float, default=None, help="Override confidence threshold")
    detect.add_argument("--stride", type=int, default=None, help="Process every Nth frame")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.command == "detect":
        config = Config.from_yaml(args.config) if args.config else Config()
        if args.model is not None:
            config.detector.model_path = args.model
        if args.conf is not None:
            config.detector.confidence = args.conf
        if args.stride is not None:
            config.video.stride = args.stride

        source = Path(args.source)
        output = (
            Path(args.output)
            if args.output
            else Path("data/processed") / f"{source.stem}_detected.mp4"
        )

        stats = run_detection(source, output, config)
        print(f"\n✅ Done → {output}")
        print(f"   frames processed  : {stats.frames}")
        print(f"   avg players/frame : {stats.avg_players_per_frame:.2f}")
        print(f"   ball visible      : {stats.ball_frames} frames ({stats.ball_visible_pct:.1f}%)")


if __name__ == "__main__":
    main()
