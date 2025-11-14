# AI Traffic Whisperer

Streamlit app using YOLOv8n to detect vehicles, count them, and recommend traffic signal duration.

## Run locally
1. python 3.9+
2. pip install -r requirements.txt
3. streamlit run traffic_ai.py

## Deploy
This repo is ready for HuggingFace Spaces (select Streamlit). First run may download yolov8n.pt.

Notes: Spaces may be CPU-only; enable GPU in Space settings for better performance.
