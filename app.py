import streamlit as st
import cv2
import json
from ultralytics import YOLO
import threading
import time

# --- Streamlit Configuration ---
st.set_page_config(page_title="High-Speed Traffic Detection", layout="wide")

# --- CLASS UNTUK BACKGROUND FRAME GRABBER ---
class VideoCaptureThread:
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        self.ret = False
        self.frame = None
        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                break
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                self.ret = True
            # Jeda agar thread grabber tidak memakan CPU berlebih
            time.sleep(0.01)

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# --- LOAD ASSETS ---
@st.cache_data
def load_data():
    with open('cctv_sources.json', 'r') as f:
        return json.load(f)

@st.cache_resource
def load_model():
    # Menggunakan model YOLO
    return YOLO("streetsOptimized.pt")

cctv_data = load_data()
model = load_model()

# --- SIDEBAR UI ---
st.sidebar.title("⚙️ Control Center")
titles = [name.replace('_', ' ').title() for name in cctv_data.keys()]
urls = list(cctv_data.values())
selected_index = st.sidebar.selectbox("📍 Select CCTV", range(len(titles)), format_func=lambda x: titles[x])
conf_threshold = st.sidebar.slider("🎯 Confidence", 0.1, 1.0, 0.45)

st.sidebar.divider()
start_btn = st.sidebar.button("▶️ START MONITORING")
stop_btn = st.sidebar.button("⏹️ STOP")

# --- MAIN DISPLAY ---
st.title("🚦 CCTV Traffic Recognition")
with st.expander("📖 User Guide: How to use this application (Click to expand)"):
    st.markdown("""
    Welcome! This application uses Artificial Intelligence (AI) to detect traffic and pedestrian in real-time via public CCTV feeds, resolution had to be scaled down due to low peformance of the host. expect inaccuracies
    Follow these simple steps to get started:
    
    1. **Select a Location**: On the left sidebar, use the **'Select CCTV Location'** dropdown menu to choose the street you want to monitor.
    2. **Adjust Sensitivity (Confidence)**: 
        *   Move the slider to the **right** (higher value) to show only objects the AI is very certain about.
        *   Move the slider to the **left** (lower value) to make the AI more sensitive, though it may occasionally misidentify objects.
    3. **Start Detection**: Click the **'▶️ Start Stream'** button to begin the live AI analysis.
    4. **Stop or Change Camera**: To stop the feed or switch to a different location, click the **'⏹️ Stop Stream'** button first.
    5. **Technical Note**: If the video freezes or fails to load, it is likely due to an unstable internet connection or the source CCTV server being temporarily offline.
    """)
st_frame = st.empty()

if start_btn:
    video_thread = VideoCaptureThread(urls[selected_index]).start()
    time.sleep(2) # Buffer awal
    
    while not stop_btn:
        ret, frame = video_thread.read()
        
        if not ret or frame is None:
            continue

        # --- OPTIMISASI AGAR TIDAK CRASH ---
        
        # 1. Perkecil resolusi input AI (Dari 640 ke 416)
        # Ini akan sangat mengurangi error "NMS time limit exceeded"
        frame_ai = cv2.resize(frame, (416, 234)) 

        # 2. Jalankan Inference dengan batasan (max_det)
        # max_det=20 membatasi AI agar tidak memproses terlalu banyak objek tumpang tindih
        results = model(frame_ai, conf=conf_threshold, verbose=False, stream=True, max_det=30)

        for r in results:
            annotated_frame = r.plot()

        # 3. Update Sintaks Streamlit terbaru (width='stretch')
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        st_frame.image(annotated_frame, channels="RGB", width='stretch')

        # 4. PENTING: Beri jeda 0.01 detik agar CPU bisa melakukan proses background Streamlit
        # Ini mencegah "Connection Reset" atau crash karena CPU 100%
        time.sleep(0.01)

    video_thread.stop()
    st.rerun() # Refresh aplikasi saat distop




