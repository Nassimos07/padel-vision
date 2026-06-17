<div align="center">

# 🎾 Padel Vision

**Broadcast-style computer vision for padel: player tracking, AR ground rings, live movement heatmaps, and foreground compositing from one match clip.**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11_kHIpcG1kUcwtzs8svAx1b2xyfl-FP7?usp=sharing)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![RF-DETR](https://img.shields.io/badge/detector-RF--DETR-0b5394.svg)](https://github.com/roboflow/rf-detr)
[![ByteTrack](https://img.shields.io/badge/tracking-ByteTrack-1b9e4c.svg)](https://github.com/ifzhang/ByteTrack)
[![Supervision](https://img.shields.io/badge/cv-Supervision-7a3ff2.svg)](https://github.com/roboflow/supervision)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

<img src="docs/ar_reel.gif" alt="Padel Vision broadcast-style AR reel" width="85%">

</div>

---

## Overview

Padel Vision turns a raw padel match video into a polished analytics film. It detects the
on-court players, keeps stable identities through rallies, projects AR rings onto the court,
builds a perspective-correct movement heatmap, and composites the real players back in front
of the graphics so the overlay feels embedded in the scene.

The easiest way to try it is the self-contained Google Colab notebook:

**[Open Padel Vision V1 in Colab](https://colab.research.google.com/drive/11_kHIpcG1kUcwtzs8svAx1b2xyfl-FP7?usp=sharing)**

Upload a clip, run the cells from top to bottom, and export a final rendered video.

<div align="center">
<img src="docs/ar_rings.jpg" alt="Perspective-projected AR ground rings" width="49%">
<img src="docs/heatmap_zonal.jpg" alt="Perspective court movement heatmap" width="49%">
</div>

## What V1 Does

- **Player detection** with RF-DETR, focused on people inside the playing area.
- **Court isolation** through a picked ROI so spectators and adjacent courts are filtered out.
- **Stable player tracking** using ByteTrack plus smoothing, coasting, and identity stabilization.
- **AR ground rings** projected onto the court plane with perspective, glow, and per-player color.
- **Foreground compositing** with YOLO segmentation so players appear in front of the graphics.
- **Movement heatmaps** mapped through a court homography so zones follow the court perspective.
- **Final film rendering** with timed transitions from live tracking to live heatmap.
- **Reusable CLI/package pieces** for calibration, detection, tracking preview, and heatmap preview.

## Tech Stack

| Layer | Tools |
| --- | --- |
| Detection | [RF-DETR](https://github.com/roboflow/rf-detr) |
| Tracking | [ByteTrack](https://github.com/ifzhang/ByteTrack) + custom stabilization |
| Segmentation | [Ultralytics YOLO11-seg](https://github.com/ultralytics/ultralytics) |
| Geometry | OpenCV homography and projected ground-plane rings |
| Rendering | OpenCV, NumPy, Supervision, ffmpeg |
| Notebook UX | Google Colab, Jupyter, ipywidgets, Matplotlib |

## Quickstart

### Option 1: Run the Notebook

1. Open the notebook in Colab:
   [Padel Vision V1](https://colab.research.google.com/drive/11_kHIpcG1kUcwtzs8svAx1b2xyfl-FP7?usp=sharing)
2. Set the runtime to GPU.
3. Run the setup cell.
4. Upload your padel clip.
5. Pick the court/ROI points, tune the overlay, and render the final film.

### Option 2: Install Locally

```bash
python3 -m venv .venv
source .venv/bin/activate

# Pick the Torch build that matches your machine/CUDA setup.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -e ".[notebook,dev]"
```

Then open:

```bash
jupyter lab notebooks/tutorials/padel_vision_v1.ipynb
```

## CLI Workflow

The package also exposes a small CLI for reusable calibration and preview steps:

```bash
padel-vision roi adjust     data/raw/match.mp4
padel-vision court adjust   data/raw/match.mp4
padel-vision court ring     data/raw/match.mp4
padel-vision detect players data/raw/match.mp4 --model nano --stride 4
padel-vision track players  data/raw/match.mp4 --model nano --stride 4
padel-vision heatmap        data/raw/match.mp4 --frame 300
```

Calibration is saved per clip under `data/calibration/<clip>.json` and reused by later commands.
Interactive commands use OpenCV windows, so they require a desktop display such as WSLg/X11.

## Project Structure

```text
padel-vision/
├── notebooks/tutorials/
│   ├── padel_vision_v1.ipynb
│   └── court_corners.txt
├── src/padel_vision/
│   ├── cli.py
│   ├── calibration.py
│   ├── detect.py
│   ├── heatmap.py
│   ├── rings.py
│   ├── track.py
│   ├── court/
│   ├── detection/
│   └── video/
├── scripts/
├── data/
├── docs/
└── tests/
```

## Roadmap

This is V1. I plan to keep improving the project weekly.

- [x] Player detection
- [x] Court/ROI calibration
- [x] Stable player IDs
- [x] AR ground rings
- [x] Foreground compositing
- [x] Movement heatmap
- [x] Final rendered film
- [ ] Ball detection and trajectory analytics
- [ ] Speed and distance in real-world units
- [ ] Shot/event detection
- [ ] Tactical dashboard and match summaries
- [ ] Cleaner production pipeline for full-match processing

## Acknowledgements

This project builds on excellent open-source tools from
[Roboflow](https://github.com/roboflow), [Ultralytics](https://github.com/ultralytics),
[Supervision](https://github.com/roboflow/supervision), and the ByteTrack community.

## License

[MIT](LICENSE) © 2026 Nassim
