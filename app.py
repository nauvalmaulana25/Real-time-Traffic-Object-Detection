import streamlit as st
import cv2
import json
from ultralytics import YOLO
import time

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Traffic Object Detection", layout="wide")

# --- CSS CUSTOMIZATION (Optional: Agar tampilan lebih rapi) ---
st.markdown("""
<style>
    div.stButton > button:first-child {
        background-color: #FF4B4B;
        color: white;
    }
    div.stButton > button:hover {
        background-color: #FF0000;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER & TITLE ---
st.title("🚦 Real-time Traffic Object Detection")
st.subheader("AI-Powered Monitoring via Public CCTV Feeds")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Configuration Panel")

# 1. Load CCTV Data
@st.cache_data
def load_data():
    try:
        # Gunakan path relatif agar aman saat deploy
        with open('cctv_sources.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("File 'cctv_sources.json' not found.")
        return {}

cctv_data = load_data()

if cctv_data:
    titles = [name.replace('_', ' ').title() for name in cctv_data.keys()]
    urls = list(cctv_data.values())
    
    selected_index = st.sidebar.selectbox("📍 Select CCTV Location", range(len(titles)), format_func=lambda x: titles[x])
else:
    st.warning("No CCTV data available.")
    titles, urls = [], []
    selected_index = 0

conf_threshold = st.sidebar.slider("🎯 Detection Confidence", 0.0, 1.0, 0.45, help="Semakin tinggi, AI semakin 'pemilih'.")

# --- OPTIMIZATION SETTINGS ---
st.sidebar.markdown("---")
st.sidebar.subheader("🚀 Performance Optimizer")
# Frame skipping: 1 = proses semua, 3 = proses 1 dari 3 frame (Hemat 66%)
frame_skip = st.sidebar.slider("Frame Skipping (Higher = Faster)", 1, 5, 3, help="Loncat frame untuk meringankan beban CPU & Internet. Nilai 3 direkomendasikan.")

st.sidebar.divider() 

# Control Buttons
col1, col2 = st.sidebar.columns(2)
start_btn = col1.button("▶️ Start", use_container_width=True)
stop_btn = col2.button("⏹️ Stop", use_container_width=True)

# 2. Load AI Model
@st.cache_resource
def load_model():
    # Pastikan menggunakan model yang ringan jika server terbatas
    return YOLO("streets.pt") 

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# 3. Video Display Area
if titles:
    st.write(f"### Currently Monitoring: **{titles[selected_index]}**")

st_frame = st.empty() # Placeholder image

# 4. Streaming Logic (Optimized)
if start_btn:
    cap = cv2.VideoCapture(urls[selected_index])
    
    if not cap.isOpened():
        st.error("❌ Connection Failed. Server CCTV mungkin offline/sibuk.")
    
    frame_count = 0
    
    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Stream buffering or reconnecting...")
            time.sleep(1) # Tunggu sebentar sebelum mencoba lagi agar tidak crash
            cap = cv2.VideoCapture(urls[selected_index]) # Reconnect logic sederhana
            continue

        frame_count += 1

        # --- OPTIMISASI UTAMA: FRAME SKIPPING ---
        # Hanya proses frame jika sisa bagi frame_count dengan frame_skip adalah 0
        if frame_count % frame_skip != 0:
            continue

        # Resize gambar agar ringan dikirim ke browser (640px lebar sudah cukup jelas)
        # Semakin kecil resolusi, semakin cepat streaming-nya.
        frame = cv2.resize(frame, (640, 360))

        # AI Inference
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # Plotting
        annotated_frame = results[0].plot()

        # Convert BGR (OpenCV) to RGB (Streamlit)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Tampilkan
        st_frame.image(annotated_frame, channels="RGB", use_container_width=True)

    cap.release()
    st.info("ℹ️ Stream stopped.")
