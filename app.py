import streamlit as st
import cv2
import json
from ultralytics import YOLO
import numpy as np

# Streamlit Page Configuration
st.set_page_config(page_title="Traffic Object Detection", layout="wide")

# HEADER & TITLE
st.title("🚦 Real-time Traffic Object Detection")
st.subheader("AI-Powered Monitoring via Public CCTV Feeds")

# (For non-technical users)
with st.expander("📖 User Guide: How to use this application (Click to expand)"):
    st.markdown("""
    Welcome! This application uses Artificial Intelligence (AI) to detect traffic objects in real-time via public CCTV feeds.
    Follow these simple steps to get started:
    
    1. **Select a Location**: On the left sidebar, use the **'Select CCTV Location'** dropdown menu to choose the street you want to monitor.
    2. **Adjust Sensitivity (Confidence)**: 
        *   Move the slider to the **right** (higher value) to show only objects the AI is very certain about.
        *   Move the slider to the **left** (lower value) to make the AI more sensitive, though it may occasionally misidentify objects.
    3. **Start Detection**: Click the **'▶️ Start Stream'** button to begin the live AI analysis.
    4. **Stop or Change Camera**: To stop the feed or switch to a different location, click the **'⏹️ Stop Stream'** button first.
    5. **CCTV Change Note**: After Switching Cameras, please click **'▶️ Start Stream'** again to initiate the new feed.            
    6. **Technical Note**: If the video freezes or fails to load, it is likely due to an unstable internet connection or the source CCTV server being temporarily offline.
    """)

# 1. Load CCTV Data
@st.cache_data
def load_data():
    # Using your specific file path
    path = r'cctv_sources.json'
    with open(path, 'r') as f:
        return json.load(f)

cctv_data = load_data()
titles = [name.replace('_', ' ').title() for name in cctv_data.keys()]
urls = list(cctv_data.values())

# 2. Sidebar Configuration
st.sidebar.header("⚙️ Configuration Panel")
selected_index = st.sidebar.selectbox("📍 Select CCTV Location", range(len(titles)), format_func=lambda x: titles[x])
conf_threshold = st.sidebar.slider("🎯 Detection Confidence", 0.0, 1.0, 0.5)

st.sidebar.divider() # Visual separator

# Control Buttons
start_btn = st.sidebar.button("▶️ Start Stream", use_container_width=True)
stop_btn = st.sidebar.button("⏹️ Stop Stream", use_container_width=True)

# 3. Load AI Model
@st.cache_resource
def load_model():
    return YOLO(r"streetsOptimized.pt")

model = load_model()

# 4. Video Display Area
st.write(f"### Currently Monitoring: **{titles[selected_index]}**")
st_frame = st.empty() # Placeholder for video frames

# 5. Streaming Loop
if start_btn:
    cap = cv2.VideoCapture(urls[selected_index])
    
    if not cap.isOpened():
        st.error("❌ Connection Failed. The CCTV server might be offline. Please try another location.")
    
    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Stream interrupted or buffering. Attempting to reconnect...")
            break

        # Resize for better web performance
        frame = cv2.resize(frame, (800, 450))

        # AI Inference
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # Draw detection boxes
        annotated_frame = results[0].plot()

        # Convert BGR to RGB for Streamlit
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display the frame
        st_frame.image(annotated_frame, channels="RGB", use_container_width=True)

    cap.release()
    st.info("ℹ️ Stream stopped by user.")

