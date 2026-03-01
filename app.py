import streamlit as st
import cv2
import numpy as np
import time
import requests
from ultralytics import YOLO
from streamlit_lottie import st_lottie

# LangChain Imports for Local Llama 3
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. PAGE CONFIG & ADVANCED UI STYLING
st.set_page_config(layout="wide", page_title="ZenUpdesh | Holistic AI", page_icon="🌿")

# Image of a high-tech meditation dashboard interface


st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
        color: #f8fafc;
    }
    
    /* Glassmorphism Cards */
    .zap-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    
    /* Neon Text and Buttons */
    h1, h2 {
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    .stButton>button {
        border-radius: 12px;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4);
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 2. ASSET LOADERS
@st.cache_resource
def load_assets():
    yolo = YOLO('yolov8n-pose.pt')
    llm = ChatOllama(model="llama3")
    lottie_url = "https://assets10.lottiefiles.com/packages/lf20_96bovdur.json"
    r = requests.get(lottie_url)
    anim = r.json() if r.status_code == 200 else None
    return yolo, llm, anim

yolo_model, llm, zen_animation = load_assets()

# 3. UTILITIES
def typewriter(text):
    container = st.empty()
    full_text = ""
    for word in text.split(" "):
        full_text += word + " "
        container.markdown(f"<div class='zap-card'><i>{full_text}</i></div>", unsafe_allow_html=True)
        time.sleep(0.05)

def draw_spiky_mandala(size, expansion, phase):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    layers, num_points = 18, 72
    theta = np.linspace(0, 2 * np.pi, num_points)
    stretch = 0.5 + (expansion * 0.4)
    
    for i in range(1, layers):
        hue = int((i / layers) * 60 + (phase * 10)) % 180
        color_bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
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

# 4. SIDEBAR & STATE
if "zen_score" not in st.session_state: st.session_state.zen_score = 0

with st.sidebar:
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    if zen_animation: st_lottie(zen_animation, height=180)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.title("Navigation")
    selection = st.selectbox("Where would you like to go?", ["👁️ Visual Flow", "🎵 Sonic Space", "📝 AI Updesh"])
    
    st.markdown("---")
    st.markdown(f"""
        <div class="zap-card" style="text-align: center;">
            <p style="margin:0; font-size: 0.9rem; opacity: 0.7;">Current Enlightenment</p>
            <h2 style="margin:0; color: #10b981 !important;">{st.session_state.zen_score} Points</h2>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("Reset Journey"):
        st.session_state.zen_score = 0
        st.rerun()

# 5. CONTENT SECTIONS
if selection == "🎵 Sonic Space":
    st.title("Sonic Space")
    st.markdown("<div class='zap-card'>Listen to 432Hz frequencies to realign your focus.</div>", unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1516280440614-37939bbacd81?auto=format&fit=crop&q=80&w=1000")

elif selection == "👁️ Visual Flow":
    st.title("Visual Regulation")
    
    col_t, col_s = st.columns([3, 1])
    with col_t:
        st.write("Sync your physical breathing with the sacred geometry below.")
    with col_s:
        run_cam = st.toggle("Activate Bio-Feedback", value=False)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<p style='text-align:center; opacity:0.6;'>Mirror Feedback</p>", unsafe_allow_html=True)
        cam_p = st.empty()
    with c2:
        st.markdown("<p style='text-align:center; opacity:0.6;'>Geometric Flow</p>", unsafe_allow_html=True)
        man_p = st.empty()

    status_container = st.empty()

    if run_cam:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        chest_pos, exp, ph, prev_y = 2.0, 0.5, 0.0, None
        last_state = "⚪ STILL"

        try:
            while run_cam:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                results = yolo_model(frame, verbose=False, stream=True)
                state = "⚪ STILL"
                
                for res in results:
                    if res.keypoints is not None and len(res.keypoints.xy) > 0:
                        kp = res.keypoints.xy[0].cpu().numpy()
                        if len(kp) > 6:
                            ls_y, rs_y = kp[5][1], kp[6][1]
                            if ls_y > 0 and rs_y > 0:
                                curr_y = (ls_y + rs_y) / 2
                                
                                if prev_y is not None:
                                    dy = curr_y - prev_y
                                    if dy < -1.0: 
                                        state = "🟢 INHALING"
                                        chest_pos += 0.25
                                        ph += 0.2
                                    elif dy > 1.0: 
                                        state = "🔵 EXHALING"
                                        chest_pos -= 0.25
                                        ph -= 0.2
                                    
                                    if last_state == "🟢 INHALING" and state == "🔵 EXHALING":
                                        st.session_state.zen_score += 1
                                        
                                    # Perform the smoothing math ONLY if prev_y is already a number
                                    prev_y = (prev_y * 0.7 + curr_y * 0.3)
                                else:
                                    # Initialize prev_y for the very first frame
                                    prev_y = curr_y
                                    
                                last_state = state

                chest_pos = np.clip(chest_pos, 0.8, 7.0)
                exp = (exp * 0.85) + ((chest_pos / 3.0) * 0.15)
                
                # Interactive Status Bar
                status_color = "#10b981" if "INHALING" in state else "#3b82f6" if "EXHALING" in state else "#64748b"
                status_container.markdown(f"""
                    <div style="background:{status_color}; padding:10px; border-radius:10px; text-align:center; font-weight:bold;">
                        {state} | Score: {st.session_state.zen_score}
                    </div>
                """, unsafe_allow_html=True)

                cam_p.image(frame, channels="BGR", use_container_width=True)
                mandala_img = draw_spiky_mandala(600, exp, ph)
                man_p.image(mandala_img, channels="BGR", use_container_width=True)
                time.sleep(0.01)
        finally:
            cap.release()

elif selection == "📝 AI Updesh":
    st.title("Consult the Sage")
    user_q = st.text_area("Share your dilemma...", placeholder="I feel overwhelmed by my workload...", height=150)

    if st.button("Invoke Wisdom"):
        if user_q:
            template = "Wise mentor using ancient tales. Dilemma: {question}. Story with moral and Option A/B."
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | llm | StrOutputParser()
            
            with st.status("Channeling the Ancients...", expanded=True):
                response = chain.invoke({"question": user_q})
            
            typewriter(response)
            
            st.markdown("### Which path will you take?")
            ca, cb = st.columns(2)
            ca.button("Path of Silence")
            cb.button("Path of Fire")