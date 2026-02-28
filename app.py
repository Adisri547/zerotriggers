import streamlit as st
import cv2
import numpy as np
import time
from ultralytics import YOLO

st.set_page_config(layout="wide")
st.title("Visual Regulation: YOLOv8 Pose Tracking")

# Load the YOLOv8 Nano Pose model (fastest version)
@st.cache_resource
def load_model():
    return YOLO('yolov8n-pose.pt')

model = load_model()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### YOLO Camera Feed")
    status_placeholder = st.empty()
    camera_feed = st.empty()

with col2:
    st.markdown("### Digital Mandala")
    mandala_feed = st.empty()

def draw_neon_mandala(size, expansion, phase):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    
    layers = 15
    num_points = 60
    theta = np.linspace(0, 2 * np.pi, num_points)
    
    # Stretch factor dictates how large the mandala gets
    stretch = 0.5 + (expansion * 0.4) 
    
    for i in range(1, layers):
        hue = int((i / layers) * 180) 
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        color = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))
        
        r1 = i * 15 * stretch
        r2 = r1 + (15 * stretch) + (10 * np.sin(phase + (i * 0.5))) 
        
        x1 = (center[0] + r1 * np.cos(theta + (phase * 0.15))).astype(int)
        y1 = (center[1] + r1 * np.sin(theta + (phase * 0.15))).astype(int)
        
        x2 = (center[0] + r2 * np.cos(theta + 0.15 + (phase * 0.15))).astype(int)
        y2 = (center[1] + r2 * np.sin(theta + 0.15 + (phase * 0.15))).astype(int)
        
        for j in range(num_points):
            pt1 = (x1[j], y1[j])
            pt2 = (x2[j], y2[j])
            cv2.line(img, pt1, pt2, color, 1, cv2.LINE_AA)
            if j > 0:
                cv2.line(img, (x1[j-1], y1[j-1]), pt1, color, 1, cv2.LINE_AA)
                
    return img

cap = cv2.VideoCapture(0)
# Lower resolution strictly to fix latency with YOLO
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

chest_position = 2.0  # Starting mandala size
expansion = 0.5       # Visual smoothing
phase = 0.0           # Animation clock
prev_y = None         # To track shoulder Y movement

state = "⚪ INITIALIZING"

while True:
    ret, frame = cap.read()
    if not ret: 
        break
        
    # Run YOLOv8 Pose prediction
    # verbose=False stops it from printing to the terminal every frame
    results = model(frame, verbose=False)
    
    # Extract keypoints
    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        
        # YOLOv8 COCO format: 5 is Left Shoulder, 6 is Right Shoulder
        if len(keypoints) > 6:
            ls_x, ls_y = int(keypoints[5][0]), int(keypoints[5][1])
            rs_x, rs_y = int(keypoints[6][0]), int(keypoints[6][1])
            
            # Draw circles on shoulders so you see what YOLO is tracking
            if ls_x != 0 and rs_x != 0:
                cv2.circle(frame, (ls_x, ls_y), 8, (0, 255, 0), -1)
                cv2.circle(frame, (rs_x, rs_y), 8, (0, 255, 0), -1)
                
                # Calculate average Y position of both shoulders
                current_y = (ls_y + rs_y) / 2
                
                if prev_y is not None:
                    # Calculate difference (delta_y)
                    # Remember: In OpenCV, Y=0 is the TOP of the image.
                    # Moving UP (Inhale) means Y decreases (delta_y is negative).
                    # Moving DOWN (Exhale) means Y increases (delta_y is positive).
                    delta_y = current_y - prev_y
                    
                    deadzone = 0.5 # Ignore tiny YOLO jitters
                    
                    if delta_y < -deadzone:
                        state = "🟢 INHALING (Expanding)"
                        chest_position += abs(delta_y) * 0.1
                        phase += 0.15 # Spin forward
                    elif delta_y > deadzone:
                        state = "🔵 EXHALING (Collapsing)"
                        chest_position -= abs(delta_y) * 0.1
                        phase -= 0.15 # Spin backward
                    else:
                        state = "⚪ STILL"
                        # Phase does not change, mandala freezes
                
                # Smooth the tracking to avoid YOLO keypoint jumping
                if prev_y is None:
                    prev_y = current_y
                else:
                    prev_y = (prev_y * 0.8) + (current_y * 0.2)

    # Prevent the mandala from shrinking to nothing or growing off-screen
    chest_position = np.clip(chest_position, 0.5, 8.0)
    
    # Smooth the visual transition
    target_expansion = chest_position / 3.0
    expansion = (expansion * 0.8) + (target_expansion * 0.2)

    # --- Visual Updates ---
    cv2.putText(frame, state, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    status_placeholder.markdown(f"**Current State:** {state}")
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    camera_feed.image(frame_rgb, use_container_width=True)
    
    mandala_img = draw_neon_mandala(600, expansion, phase)
    mandala_rgb = cv2.cvtColor(mandala_img, cv2.COLOR_BGR2RGB)
    mandala_feed.image(mandala_rgb, use_container_width=True)
    
    time.sleep(0.01)

cap.release()