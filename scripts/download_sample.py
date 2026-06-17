"""Download a sample clip into data/raw/ (you supply a URL you have rights to use).

Usage:
    python scripts/download_sample.py "<video-url>"

Requires yt-dlp:  pip install yt-dlp
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1

    url = sys.argv[1]
    if shutil.which("yt-dlp") is None:
        print("yt-dlp not found. Install it with:  pip install yt-dlp")
        return 1

    print(
        "⚠️  Only download clips you have the rights to use publicly "
        "(this project is meant for sharing). Avoid copyrighted broadcast footage.\n"
    )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio/best[height<=720]",
        "-o", str(RAW_DIR / "%(title).50s.%(ext)s"),
        url,
    ]
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
