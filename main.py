import streamlit as st
import cv2
import numpy as np
import time
import os
import requests
from ultralytics import YOLO
from streamlit_lottie import st_lottie

# LangChain Imports for Local Llama 3
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. PAGE CONFIG & UI STYLING
st.set_page_config(layout="wide", page_title="ZenUpdesh | Holistic AI", page_icon="🌿")

# Custom CSS for Glassmorphism and Neon accents
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stSidebar { background-color: #161b22 !important; border-right: 1px solid #30363d; }
    h1, h2, h3 { color: #58a6ff !important; font-family: 'Helvetica Neue', sans-serif; }
    .stButton>button {
        width: 100%;
        border-radius: 25px;
        background: linear-gradient(45deg, #00d2ff 0%, #3a7bd5 100%);
        color: white; font-weight: bold; border: none; transition: 0.3s;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 4px 15px rgba(0, 210, 255, 0.4); }
    .stTextArea textarea { background-color: #0d1117; border: 1px solid #30363d; color: #c9d1d9; border-radius: 10px; }
    [data-testid="stMetricValue"] { color: #79fe9d !important; }
    </style>
    """, unsafe_allow_html=True)

# 2. ASSET LOADERS
def load_lottie(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

@st.cache_resource
def load_yolo():
    return YOLO('yolov8n-pose.pt')

@st.cache_resource
def load_llm():
    return ChatOllama(model="llama3")

yolo_model = load_yolo()
llm = load_llm()
zen_animation = load_lottie("https://assets10.lottiefiles.com/packages/lf20_96bovdur.json")

# 3. HELPER FUNCTIONS
def typewriter(text):
    container = st.empty()
    full_text = ""
    for word in text.split(" "):
        full_text += word + " "
        container.markdown(f"*{full_text}*")
        time.sleep(0.05)

def draw_spiky_mandala(size, expansion, phase):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    layers, num_points = 18, 72
    theta = np.linspace(0, 2 * np.pi, num_points)
    stretch = 0.5 + (expansion * 0.4)
    
    for i in range(1, layers):
        hue = int((i / layers) * 180)
        color_bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        color = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))
        
        r1 = i * 14 * stretch
        spike = 1.0 if i % 2 == 0 else 1.2
        r2 = (r1 + (15 * stretch)) * spike + (8 * np.sin(phase + (i * 0.8)))
        
        x1 = (center[0] + r1 * np.cos(theta + (phase * 0.1))).astype(int)
        y1 = (center[1] + r1 * np.sin(theta + (phase * 0.1))).astype(int)
        x2 = (center[0] + r2 * np.cos(theta + 0.08 + (phase * 0.1))).astype(int)
        y2 = (center[1] + r2 * np.sin(theta + 0.08 + (phase * 0.1))).astype(int)
        
        for j in range(num_points):
            cv2.line(img, (x1[j], y1[j]), (x2[j], y2[j]), color, 1, cv2.LINE_AA)
            if j > 0:
                cv2.line(img, (x1[j-1], y1[j-1]), (x1[j], y1[j]), color, 1, cv2.LINE_AA)
    return img

# 4. SIDEBAR NAVIGATION & METRICS
if "zen_score" not in st.session_state: st.session_state.zen_score = 0

with st.sidebar:
    if zen_animation: st_lottie(zen_animation, height=150)
    st.title("ZenNav")
    selection = st.radio("Journey Path:", ["🎵 Sonic Space", "👁️ Visual Flow", "📝 AI Updesh"])
    st.markdown("---")
    st.metric("Zen Points", f"{st.session_state.zen_score} 🧘")
    st.caption("Complete breath cycles to earn points.")

# 5. SECTION 1: SONIC
if selection == "🎵 Sonic Space":
    st.title("Sonic Healing")
    st.write("Tune into frequencies designed to stabilize the heart rate.")
    st.image("https://images.unsplash.com/photo-1516280440614-37939bbacd81?auto=format&fit=crop&q=80&w=1000")

# 6. SECTION 2: VISUAL FLOW (YOLO + MANDALA)
elif selection == "👁️ Visual Flow":
    st.title("Visual Regulation")
    st.write("Sync your physical movements with digital geometry.")
    
    run_cam = st.toggle("Activate Bio-Feedback", value=False)
    
    c1, c2 = st.columns([1, 1])
    with c1:
        status_box = st.info("Stand in view to calibrate...")
        cam_p = st.empty()
    with c2:
        man_p = st.empty()

    if run_cam:
        cap = cv2.VideoCapture(0)
        chest_pos, exp, ph, prev_y = 2.0, 0.5, 0.0, None
        last_state = ""

        while run_cam:
            ret, frame = cap.read()
            if not ret: break
            results = yolo_model(frame, verbose=False)
            
            state = "⚪ STILL"
            if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                kp = results[0].keypoints.xy[0].cpu().numpy()
                if len(kp) > 6:
                    curr_y = (kp[5][1] + kp[6][1]) / 2
                    if prev_y is not None:
                        dy = curr_y - prev_y
                        if dy < -0.6: 
                            state = "🟢 INHALING"; chest_pos += 0.15; ph += 0.2
                        elif dy > 0.6: 
                            state = "🔵 EXHALING"; chest_pos -= 0.15; ph -= 0.2
                        
                        # Point Logic: Inhale to Exhale transition
                        if last_state == "🟢 INHALING" and state == "🔵 EXHALING":
                            st.session_state.zen_score += 1
                            
                    prev_y = curr_y if prev_y is None else (prev_y * 0.8 + curr_y * 0.2)
                    last_state = state

            chest_pos = np.clip(chest_pos, 0.8, 7.0)
            exp = (exp * 0.8) + ((chest_pos / 3.0) * 0.2)
            
            status_box.markdown(f"### Mode: {state}")
            cam_p.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            man_p.image(cv2.cvtColor(draw_spiky_mandala(600, exp, ph), cv2.COLOR_BGR2RGB), use_container_width=True)
            time.sleep(0.01)
        cap.release()

# 7. SECTION 3: AI UPDESH (NARRATIVE)
elif selection == "📝 AI Updesh":
    st.title("Narrative Therapy (AI Updesh)")
    st.write("The Sage translates your modern stress into ancient metaphors.")
    
    user_q = st.text_area("What is your dilemma?", height=120, placeholder="Example: I am worried about my future...")

    if st.button("Consult the Sage"):
        if user_q:
            template = """
            You are a wise mentor using Panchatantra/Jataka tales.
            Dilemma: {question}
            
            1. Narrate an ancient story where the user is the protagonist.
            2. Integrate the moral.
            3. End with a path choice: 'Option A' or 'Option B'.
            """
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | llm | StrOutputParser()
            
            with st.status("Consulting Ancient Archives...", expanded=True) as status:
                st.write("Recalling fables...")
                response = chain.invoke({"question": user_q})
                status.update(label="Wisdom Found!", state="complete", expanded=False)
            
            st.markdown("---")
            typewriter(response)
            
            st.markdown("### Choose your destiny:")
            col_a, col_b = st.columns(2)
            col_a.button("The Path of Patience")
            col_b.button("The Path of Action")
        else:
            st.warning("Please share your thoughts first.")