import streamlit as st
import cv2
import json
from ultralytics import YOLO
import threading
import time

# --- Streamlit Configuration ---
st.set_page_config(page_title="High-Speed Traffic Detection", layout="wide")

# --- CUSTOM CSS FOR SMOOTHNESS ---
st.markdown("<style>img { border-radius: 10px; }</style>", unsafe_allow_html=True)

# --- CLASS UNTUK BACKGROUND FRAME GRABBER ---
class VideoCaptureThread:
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        self.ret = False
        self.frame = None
        self.stopped = False
        # Thread untuk membaca frame
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                break
            self.ret, self.frame = self.cap.read()
            # Sedikit jeda agar tidak menghabiskan CPU untuk pembacaan saja
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
    return YOLO("streets.pt")

cctv_data = load_data()
model = load_model()

# --- SIDEBAR UI ---
st.sidebar.title("⚙️ Control Center")
titles = [name.replace('_', ' ').title() for name in cctv_data.keys()]
urls = list(cctv_data.values())
selected_index = st.sidebar.selectbox("📍 Select CCTV", range(len(titles)), format_func=lambda x: titles[x])
conf_threshold = st.sidebar.slider("🎯 Confidence", 0.0, 1.0, 0.4)

st.sidebar.divider()
start_btn = st.sidebar.button("▶️ START MONITORING", use_container_width=True)
stop_btn = st.sidebar.button("⏹️ STOP", use_container_width=True)

# --- MAIN DISPLAY ---
st.title("🚦 Traffic Object Recognition")
st_frame = st.empty()

if start_btn:
    # Inisialisasi Thread pembaca video
    video_thread = VideoCaptureThread(urls[selected_index]).start()
    
    # Beri waktu buffer awal
    time.sleep(2)
    
    # Placeholder untuk FPS
    fps_display = st.sidebar.empty()
    prev_time = 0

    while not stop_btn:
        ret, frame = video_thread.read()
        
        if not ret or frame is None:
            st_frame.warning("Connecting to CCTV stream... please wait.")
            continue

        # --- OPTIMIZATION STEPS ---
        # 1. Resize ke ukuran kecil untuk diproses AI (Sangat penting!)
        # YOLOv8 paling cepat pada 640px
        frame_ai = cv2.resize(frame, (640, 360))

        # 2. AI Inference
        # verbose=False mengurangi beban printing di terminal
        results = model(frame_ai, conf=conf_threshold, verbose=False, stream=True)

        # 3. Visualization
        for r in results:
            annotated_frame = r.plot()

        # 4. Convert format warna
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # 5. Display ke Streamlit
        st_frame.image(annotated_frame, channels="RGB", use_container_width=True)

        # 6. Hitung FPS Real-time
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        fps_display.metric("Performance", f"{int(fps)} FPS")

    video_thread.stop()
    st.success("Monitoring stopped.")

