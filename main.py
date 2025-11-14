# traffic_ai.py
# Deploy-ready Streamlit app for HuggingFace Spaces / Streamlit Cloud
# Run locally: streamlit run traffic_ai.py

import os
import tempfile
import time
from collections import deque

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="AI Traffic Whisperer", page_icon="🚦", layout="wide")

# Minimalist styling
st.markdown(
    """
    <style>
      .title {font-size:28px; font-weight:700; margin-bottom:6px;}
      .muted {color: #9aa0a6; margin-bottom:20px;}
      .card {background:#0f1724; padding:12px; border-radius:8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">🚦 AI Traffic Whisperer</div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Upload traffic video or enter CCTV RTSP URL. Uses YOLOv8n (yolov8n.pt).</div>', unsafe_allow_html=True)

# Settings
TARGET_CLASSES = {"car", "motorcycle", "bus", "truck"}
DEFAULT_THRESHOLD = 5

with st.sidebar:
    st.header("Controls")
    input_mode = st.radio("Input", ["Upload Video", "CCTV/RTSP (URL)"], index=0)
    vehicle_threshold = st.slider("Vehicles > N → Green", 1, 20, DEFAULT_THRESHOLD)
    st.markdown("---")
    st.caption("Note: Spaces/Cloud may be CPU-only. First run may download model weights.")

# Load model (cached)
@st.cache_resource
def get_model():
    return YOLO("yolov8n.pt")

model = get_model()

# Helper functions
def adaptive_timing(count: int, threshold: int):
    if count > threshold:
        extra = max(0, count - threshold)
        duration = min(10 + (extra // 2) * 2, 60)
        return "GREEN", duration
    else:
        return "RED", 5

def process_frame(frame_bgr, model, threshold):
    # Resize for speed
    frame_bgr = cv2.resize(frame_bgr, (640, 480))
    results = model.predict(frame_bgr, conf=0.25, verbose=False)[0]

    count = 0
    names = results.names
    boxes = getattr(results, "boxes", None)
    if boxes is not None and getattr(boxes, "xyxy", None) is not None:
        for xyxy, cls in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy()):
            cname = names.get(int(cls), str(int(cls)))
            if cname in TARGET_CLASSES:
                x1, y1, x2, y2 = map(int, xyxy)
                count += 1
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (60, 200, 255), 2)
                cv2.putText(frame_bgr, cname, (x1, max(18, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

    decision, duration = adaptive_timing(count, threshold)
    color = (0,200,0) if decision == "GREEN" else (0,0,255)
    cv2.rectangle(frame_bgr, (0,0), (640,40), (0,0,0), -1)
    cv2.putText(frame_bgr, f"Vehicles: {count}", (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(frame_bgr, f"Signal: {decision} ({duration}s)", (320,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame_bgr, count, decision, duration

# UI placeholders
col1, col2 = st.columns([3,1])
with col2:
    st.markdown("### Live Stats")
    stat_status = st.empty()
    stat_count = st.empty()
    stat_avg = st.empty()
with col1:
    video_box = st.empty()

# Processing loop helpers
def run_capture_loop(cap):
    stats_window = deque(maxlen=int(24*60))  # last 60 seconds @ ~24fps
    last_update = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start = time.time()
        annotated, count, decision, duration = process_frame(frame, model, vehicle_threshold)
        stats_window.append(count)
        avg = float(np.mean(stats_window)) if stats_window else 0.0

        # update UI (convert BGR->RGB)
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        video_box.image(frame_rgb, channels="RGB", use_column_width=True)

        # stats
        icon = "🟢" if decision == "GREEN" else "🔴"
        stat_status.markdown(f"**Signal:** {icon} **{decision}** ({duration}s)")
        stat_count.markdown(f"**Current Vehicles:** {count}")
        stat_avg.markdown(f"**Avg (approx, last 60s):** {avg:.2f}")

        # throttle UI updates (helps with slow CPUs/cloud)
        elapsed = time.time() - start
        if elapsed < 1/10:
            time.sleep(max(0, 1/10 - elapsed))  # target ~10 FPS UI updates

# Input handling
if input_mode == "Upload Video":
    uploaded = st.file_uploader("Upload .mp4 or .avi", type=["mp4", "avi"])
    if uploaded is not None:
        # write to temp file (reliable)
        suffix = os.path.splitext(uploaded.name)[1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.read())
        tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        run_capture_loop(cap)
        cap.release()
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
else:
    rtsp_url = st.text_input("Enter RTSP/HTTP camera URL (or leave blank):")
    start_stream = st.button("Start Stream")
    if start_stream and rtsp_url:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            st.error("Could not open stream. Check URL & network.")
        else:
            run_capture_loop(cap)
            cap.release()
    elif start_stream:
        st.info("Paste a working RTSP/HTTP URL before starting.")

st.markdown("---")
st.caption("If the app is slow on Spaces, enable GPU in Space settings or test locally with GPU.")
