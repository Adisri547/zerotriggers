import streamlit as st
import cv2
import numpy as np
import time
import requests
import os
import soundfile as sf
import datetime
import pygame
import threading
from ultralytics import YOLO
from streamlit_lottie import st_lottie
from deepface import DeepFace

# LangChain Imports for Local Llama 3
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. PAGE CONFIG & UI STYLING
st.set_page_config(layout="wide", page_title="ZenUpdesh | Holistic AI", page_icon="🌿")

# --- FIXED INITIALIZATION BLOCK ---
if 'audio_setup' not in st.session_state:
    try:
        pygame.mixer.init()
    except:
        pass
    st.session_state.audio_setup = True
    st.session_state.is_running_audio = False
    st.session_state.current_track = "None"
    st.session_state.detected_emotion = "NEUTRAL"
    st.session_state.zen_score = 0
    st.session_state.is_stressed = False
    st.session_state.emotion = "neutral"

# NEW THEME: "Vedic Parchment" CSS
st.markdown("""
    <style>
    .stApp { 
        background-color: #fdf5e6; 
        background-image: url("https://www.transparenttextures.com/patterns/natural-paper.png");
        color: #4a3728; 
    }
    
    .zap-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 25px;
        padding: 25px;
        border: 2px solid #c19a6b;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    
    .zap-card:hover {
        transform: translateY(-5px);
        border-color: #8b4513;
        background: rgba(255, 255, 255, 0.9);
    }

    .prahar-badge {
        background: #8b4513;
        color: #fdf5e6;
        padding: 5px 15px;
        border-radius: 50px;
        font-weight: bold;
        font-size: 0.8rem;
    }

    h1, h2, h3 { 
        color: #5d4037 !important; 
        font-family: 'Garamond', serif;
    }

    .stButton>button {
        border-radius: 20px;
        background: #8b4513;
        color: #fdf5e6;
        border: 2px solid #5d4037;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. ASSET LOADERS & HEALING LOGIC
@st.cache_resource
def load_assets():
    yolo = YOLO('yolov8n-pose.pt')
    llm = ChatOllama(model="llama3")
    lottie_url = "https://assets10.lottiefiles.com/packages/lf20_96bovdur.json"
    try:
        r = requests.get(lottie_url)
        anim = r.json() if r.status_code == 200 else None
    except:
        anim = None
    return yolo, llm, anim

yolo_model, llm, zen_animation = load_assets()

def get_target_track(emo):
    hour = datetime.datetime.now().hour
    emo = emo.lower()
    if emo in ['sad', 'angry', 'fear', 'disgust']:
        return "todi.mp3" if 4 <= hour < 12 else "bhairavi.mp3"
    return "ragi.mp3" 

def run_audio_engine():
    while st.session_state.get('is_running_audio', False):
        emo = st.session_state.get('detected_emotion', 'NEUTRAL')
        target = get_target_track(emo)
        if target != st.session_state.get('current_track'):
            if os.path.exists(target):
                try:
                    pygame.mixer.music.load(target)
                    pygame.mixer.music.play(-1)
                    st.session_state.current_track = target
                except:
                    pass
        time.sleep(1)

# 3. UTILITIES
def typewriter(text):
    container = st.empty()
    full_text = ""
    for word in text.split(" "):
        full_text += word + " "
        container.markdown(f"<div class='zap-card'><i>{full_text}</i></div>", unsafe_allow_html=True)
        time.sleep(0.05)

def draw_spiky_mandala(size, expansion, phase):
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    center = (size // 2, size // 2)
    layers, num_points = 18, 72
    theta = np.linspace(0, 2 * np.pi, num_points)
    stretch = 0.5 + (expansion * 0.4)
    for i in range(1, layers):
        hue = int((i / layers) * 60 + (phase * 10)) % 180
        color_bgr = cv2.cvtColor(np.uint8([[[hue, 200, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
        color = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))
        r1 = i * 14 * stretch
        r2 = (r1 + (15 * stretch)) * (1.1 if i % 2 == 0 else 1.3) + (8 * np.sin(phase + (i * 0.8)))
        rot = phase * 0.1
        x1 = (center[0] + r1 * np.cos(theta + rot)).astype(int)
        y1 = (center[1] + r1 * np.sin(theta + rot)).astype(int)
        x2 = (center[0] + r2 * np.cos(theta + 0.08 + rot)).astype(int)
        y2 = (center[1] + r2 * np.sin(theta + 0.08 + rot)).astype(int)
        for j in range(num_points):
            cv2.line(img, (x1[j], y1[j]), (x2[j], y2[j]), color, 1, cv2.LINE_AA)
    return img

# 4. SIDEBAR NAVIGATION
with st.sidebar:
    if zen_animation: st_lottie(zen_animation, height=180)
    st.title("Abhiyan (Journey)")
    selection = st.selectbox("Select Pathway", ["🏠 Home: Swagat", "🎵 Raag-Rasa", "☸️ Prana-Yantra", "📖 Katha-Bodhi"])
    
    score = st.session_state.get('zen_score', 0)
    st.markdown(f"""<div class="zap-card" style="text-align:center;">Punya (Enlightenment): <b>{score}</b></div>""", unsafe_allow_html=True)
    
    if st.button("Punarjanma (Reset)"):
        st.session_state.zen_score = 0
        st.session_state.is_stressed = False
        st.session_state.is_running_audio = False
        st.session_state.detected_emotion = "NEUTRAL"
        pygame.mixer.music.stop()
        st.rerun()

# 5. CONTENT SECTIONS
if selection == "🏠 Home: Swagat":
    st.title("नमस्ते | Welcome to ZenUpdesh")
    
    # INTERACTIVE PRAHAR CLOCK
    now = datetime.datetime.now()
    hour = now.hour
    if 4 <= hour < 12: prahar = "Pratah (Morning)"; raga = "Todi"
    elif 17 <= hour < 24: prahar = "Sayankala (Evening)"; raga = "Bhairavi"
    else: prahar = "Madhya (Neutral)"; raga = "Yaman"

    st.markdown(f"""
        <div style="text-align:center; padding: 10px;">
            <span class="prahar-badge">Current Prahar: {prahar}</span>
            <span class="prahar-badge" style="background:#c19a6b;">Active Healing: {raga}</span>
        </div>
    """, unsafe_allow_html=True)

    st.write("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div class="zap-card">
                <h3>🕉️ Holistic Synthesis</h3>
                <p>Experience the union of <b>Ancient Vedic Wisdom</b> and <b>Generative AI</b>. 
                Our system uses bio-feedback to harmonize your internal Rasas.</p>
                <p><i>"Sound is the medicine of the soul."</i></p>
            </div>
            <div class="zap-card">
                <h3>⚡ Real-time Regulation</h3>
                <p>We monitor your micro-expressions and breathing patterns to dynamically 
                adjust the digital environment.</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.write("### Interactive Journey Guide")
        tab1, tab2, tab3 = st.tabs(["Sonic", "Visual", "Wisdom"])
        with tab1:
            st.info("**Raag-Rasa**: Uses Computer Vision to detect stress and plays therapeutic Raagas based on the Prahar.")
        with tab2:
            st.info("**Prana-Yantra**: An interactive Mandala that expands and contracts with your breath (YOLO Pose detection).")
        with tab3:
            st.info("**Katha-Bodhi**: A conversational AI mentor trained to provide ethical guidance through ancient parables.")

    st.divider()
    st.caption("ZenUpdesh v2.0 | Traditional Roots, Digital Wings")

elif selection == "🎵 Raag-Rasa":
    st.title("Raag-Rasa")
    col_cam, col_ctrl = st.columns([2, 1])
    with col_ctrl:
        st.subheader("Controls")
        if st.button("▶ START AI HEALING", use_container_width=True):
            st.session_state.is_running_audio = True
            threading.Thread(target=run_audio_engine, daemon=True).start()
        if st.button("⏹ STOP & PLAY LAST", use_container_width=True):
            st.session_state.is_running_audio = False
            last_emo = st.session_state.get('detected_emotion', 'NEUTRAL')
            final_track = get_target_track(last_emo)
            if os.path.exists(final_track):
                try:
                    pygame.mixer.music.load(final_track)
                    pygame.mixer.music.play(-1)
                    st.session_state.current_track = final_track
                    st.success(f"Final therapy: {last_emo.upper()}")
                except: pass
        if st.button("🔇 FULL RESET", use_container_width=True):
            st.session_state.is_running_audio = False
            st.session_state.current_track = "None"
            pygame.mixer.music.stop()
            st.rerun()
        st.divider()
        current_emo = st.session_state.get('detected_emotion', 'NEUTRAL')
        st.metric("Rasa (Emotion)", current_emo.upper())
        st.write(f"Healing Track: **{st.session_state.get('current_track', 'None')}**")
    with col_cam:
        if st.session_state.get('is_running_audio', False):
            cap = cv2.VideoCapture(0)
            frame_placeholder = st.empty()
            while st.session_state.get('is_running_audio', False):
                ret, frame = cap.read()
                if not ret: break
                try:
                    res = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                    st.session_state.detected_emotion = res[0]['dominant_emotion']
                    cv2.putText(frame, f"Emotion: {st.session_state.detected_emotion.upper()}", 
                                (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except: pass
                frame_placeholder.image(frame, channels="BGR", use_container_width=True)
                time.sleep(0.01)
            cap.release()
        else:
            st.info("Start the AI to begin sonic therapy.")

# (Remaining sections Prana-Yantra and Katha-Bodhi remain unchanged)
elif selection == "☸️ Prana-Yantra":
    st.title("Prana-Yantra")
    col_t, col_s = st.columns([3, 1])
    with col_t: st.write("Sync breath with geometry. AI monitors micro-expressions.")
    with col_s: run_cam = st.toggle("Activate Feedback", value=False)
    cam_p, man_p = st.columns(2)
    cam_display, man_display = cam_p.empty(), man_p.empty()
    status_container = st.empty()
    if run_cam:
        cap = cv2.VideoCapture(0)
        chest_pos, exp, ph, prev_y, frame_counter = 2.0, 0.5, 0.0, None, 0
        try:
            while run_cam:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                if frame_counter % 30 == 0:
                    try:
                        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                        st.session_state.emotion = result[0]['dominant_emotion']
                    except: pass
                results = yolo_model(frame, verbose=False, stream=True)
                state = "⚪ STILL"
                for res in results:
                    if res.keypoints is not None and len(res.keypoints.xy) > 0:
                        kp = res.keypoints.xy[0].cpu().numpy()
                        if len(kp) > 6:
                            curr_y = (kp[5][1] + kp[6][1]) / 2
                            if prev_y is not None:
                                dy = curr_y - prev_y
                                if dy < -1.0: state = "🟢 INHALING"; chest_pos += 0.2; ph += 0.2
                                elif dy > 1.0: state = "🔵 EXHALING"; chest_pos -= 0.2; ph -= 0.2
                                prev_y = (prev_y * 0.7 + curr_y * 0.3)
                            else: prev_y = curr_y
                chest_pos = np.clip(chest_pos, 0.8, 7.0)
                exp = (exp * 0.85) + ((chest_pos / 3.0) * 0.15)
                man_display.image(draw_spiky_mandala(600, exp, ph), channels="BGR", use_container_width=True)
                cam_display.image(frame, channels="BGR", use_container_width=True)
                frame_counter += 1
                time.sleep(0.01)
        finally:
            cap.release()

elif selection == "📖 Katha-Bodhi":
    st.title("Katha-Bodhi")
    user_q = st.text_area("Share your dilemma...", height=150)
    if st.button("Invoke Wisdom"):
        if user_q:
            template = "Wise mentor using ancient tales. Dilemma: {question}."
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | llm | StrOutputParser()
            with st.status("Gathering Insight..."):
                response = chain.invoke({"question": user_q})
            typewriter(response)