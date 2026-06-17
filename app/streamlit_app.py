"""Interactive padel analytics dashboard (Stage 1: detection).

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from padel_analytics.config import Config
from padel_analytics.pipeline import run_detection
from padel_analytics.video.io import to_h264

st.set_page_config(page_title="Padel Match Analytics", page_icon="🎾", layout="wide")

st.title("🎾 Padel Match Analytics")
st.caption("Stage 1 — Object detection of players & ball (RF-DETR / YOLOv11)")

with st.sidebar:
    st.header("⚙️ Settings")
    backend = st.radio("Detector", ["RF-DETR", "YOLO"], horizontal=True)
    if backend == "RF-DETR":
        backend_key = "rfdetr"
        model = st.selectbox("RF-DETR model", ["nano", "small", "medium", "large"], index=2)
    else:
        backend_key = "yolo"
        model = st.selectbox("YOLO model", ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"], index=2)
    conf = st.slider("Confidence", 0.05, 0.90, 0.50, 0.05)
    stride = st.slider("Frame stride (higher = faster)", 1, 5, 1)
    detect_ball = st.checkbox("Detect ball (experimental)", value=False)
    st.markdown("---")
    st.caption("Built with RF-DETR (Roboflow) + Supervision")

uploaded = st.file_uploader("Upload a short padel clip", type=["mp4", "mov", "avi"])

if uploaded is None:
    st.info("👆 Upload a 10–30s clip to see players and the ball detected frame by frame.")
    st.stop()

st.video(uploaded)

if not st.button("▶️ Run detection", type="primary"):
    st.stop()

with tempfile.TemporaryDirectory() as tmp:
    src = Path(tmp) / uploaded.name
    src.write_bytes(uploaded.getvalue())
    out = Path(tmp) / "detected.mp4"

    config = Config()
    config.detector.backend = backend_key
    config.detector.model = model
    config.detector.confidence = conf
    config.detector.detect_ball = detect_ball
    config.video.stride = stride

    with st.spinner("Detecting players & ball… (first run downloads the model)"):
        stats = run_detection(src, out, config, progress=False)
        playable = to_h264(out, Path(tmp) / "detected_h264.mp4")

    st.subheader("Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Frames analyzed", stats.frames)
    c2.metric("Avg players / frame", f"{stats.avg_players_per_frame:.2f}")
    c3.metric("Ball visible", f"{stats.ball_visible_pct:.0f}%")

    st.video(playable.read_bytes())
    st.success("Done! This is Stage 1 — tracking, minimap and stats come next.")
