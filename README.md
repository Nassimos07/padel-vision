<div align="center">

# 🎾 Padel Match Analytics

**Computer-vision analytics for padel — detect, track, and analyze players and the ball straight from match video.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![RF-DETR](https://img.shields.io/badge/detector-RF--DETR-0b5394.svg)](https://github.com/roboflow/rf-detr)
[![Supervision](https://img.shields.io/badge/annotations-Supervision-7a3ff2.svg)](https://github.com/roboflow/supervision)

<!-- Add your demo GIF here once Stage 1 runs: docs/demo.gif -->
<img src="docs/demo.gif" alt="Padel detection demo" width="80%">

</div>

---

## 📌 What is this?

A step-by-step computer-vision pipeline that turns a raw padel match video into rich analytics.
We build it **feature by feature** — each stage is a self-contained, reviewable module.

> **Stage 1 (current): object detection** of the players with RF-DETR. _(Ball detection is
> deferred — a tiny, fast object that needs a dedicated model; see the roadmap.)_

## ✨ Features & Roadmap

- [x] **Stage 1 — Object detection** · players, annotated overlay video _(RF-DETR)_
- [ ] **Stage 2 — Multi-object tracking** · persistent player IDs + ball trajectory _(ByteTrack)_
- [ ] **Stage 3 — Court detection & homography** · top-down minimap
- [ ] **Stage 4 — Player movement** · heatmaps + distance covered
- [ ] **Stage 5 — Ball detection & analytics** · dedicated ball model + trajectory/speed
- [ ] **Stage 6 — Shot/event detection** · bandeja, víbora, smash, volley
- [ ] **Stage 7 — Match stats dashboard** · the full interactive story
- [ ] **Stage 8 — (optional) outcome model** · the MLOps cherry on top

## 🧱 Tech stack

| Area | Tools |
|------|-------|
| Detection | [RF-DETR](https://github.com/roboflow/rf-detr) (players) · [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) (swappable backend) |
| CV utilities | [Supervision](https://github.com/roboflow/supervision), OpenCV, ffmpeg |
| Analytics | NumPy, pandas |
| Visualization | Plotly, Streamlit, Matplotlib/Seaborn |
| Quality | ruff, black, pytest, pre-commit |

## 📂 Project structure

```
padel-analytics/
├── src/padel_analytics/      # the package
│   ├── config.py             # typed config (YAML-backed)
│   ├── pipeline.py           # end-to-end runners
│   ├── detection/            # detector (YOLO) + annotation
│   └── video/                # video I/O helpers
├── app/streamlit_app.py      # interactive dashboard
├── notebooks/                # narrative analysis notebooks
├── config/default.yaml       # default settings
├── data/{raw,processed}/     # videos in / results out (gitignored)
├── models/                   # weights (gitignored)
├── scripts/                  # helper scripts
└── tests/                    # unit tests
```

## 🚀 Quickstart

```bash
# 1. Create an isolated environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install PyTorch with the right CUDA build (WSL2 / CUDA 12.x example)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install the project (+ app, notebook, dev extras)
pip install -e ".[app,notebook,dev]"

# 4. Drop a padel clip into data/raw/, then run detection
padel detect data/raw/match.mp4

# 5. Or launch the interactive app
streamlit run app/streamlit_app.py
```

> **No GPU?** It runs on CPU too — just slower. Use `--model yolo11n.pt` for a lighter model.

## 🖥️ Usage

```bash
# Basic
padel detect data/raw/match.mp4

# Choose RF-DETR model size + confidence, custom output path
padel detect data/raw/match.mp4 --model small --conf 0.4 -o data/processed/out.mp4

# Swap the detector backend to YOLO
padel detect data/raw/match.mp4 --backend yolo --model yolo11m.pt

# Use a config file
padel detect data/raw/match.mp4 -c config/default.yaml
```

## 🤝 Acknowledgements

Built on the excellent open-source work of [Roboflow RF-DETR](https://github.com/roboflow/rf-detr),
[Ultralytics](https://github.com/ultralytics/ultralytics) and
[Roboflow Supervision](https://github.com/roboflow/supervision).

## 📄 License

[MIT](LICENSE) © 2026 Nassim
