"""
Helmet Detection & Road Safety Analyzer
Main application entry point.
Decoupled modular architecture with Apple-inspired light neutral UI/UX and zero emojis.
"""

import os
import time
import uuid
import queue
import shutil
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# UI Theme & Presentation Modules
from ui.theme import load_theme
from ui.layout import (
    render_top_header,
    render_sidebar_controls,
    render_live_detection_status,
    render_footer
)
from ui.pages.image import render_single_image_page
from ui.pages.batch import render_batch_page
from ui.pages.video import render_video_page
from ui.pages.live_camera import render_live_camera_page
from ui.pages.analytics import render_analytics_page

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Helmet Detection & Road Safety Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# PATH DEFINITIONS & DIRECTORY INITIALIZATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model weights: prefer models/best.pt (canonical deployment path),
# fall back to legacy runs/train path for backwards compatibility.
_MODEL_CANONICAL = os.path.join(BASE_DIR, "models", "best.pt")
_MODEL_LEGACY = os.path.join(BASE_DIR, "runs", "train", "helmet_detector2", "weights", "best.pt")
MODEL_PATH = _MODEL_CANONICAL if os.path.exists(_MODEL_CANONICAL) else _MODEL_LEGACY

SAMPLE_DIR = os.path.join(BASE_DIR, "sample_media")
SAMPLE_IMAGE_PATH = os.path.join(SAMPLE_DIR, "sample_street.jpg")
SAMPLE_VIDEO_PATH = os.path.join(SAMPLE_DIR, "sample_traffic.mp4")

os.makedirs(SAMPLE_DIR, exist_ok=True)

# 4 Trained YOLOv8 Classes
CLASS_NAMES = {0: "bicyclist", 1: "driver", 2: "helmet", 3: "no-helmet"}

# ==============================================================================
# MODEL CACHING — loaded once per server process, shared across all sessions
# ==============================================================================
@st.cache_resource(show_spinner=False)
def get_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Trained YOLOv8 model weights not found at: {path}\n"
            "Place best.pt in the models/ directory or at "
            "runs/train/helmet_detector2/weights/best.pt"
        )
    return YOLO(path)

model = get_model(MODEL_PATH)

# ==============================================================================
# INJECT DESIGN SYSTEM STYLESHEET
# ==============================================================================
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "light"
load_theme(st.session_state.theme_mode)

# ==============================================================================
# SESSION-SCOPED ISOLATION
# Each browser session gets its own:
#   - violation snapshot directory  (prevents cross-user file exposure)
#   - WebRTC event queue            (prevents cross-user stat bleed)
# Both are stored in st.session_state so they are private to each tab/user.
# ==============================================================================
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

SESSION_VIOLATION_DIR = os.path.join(
    BASE_DIR, "violations", st.session_state.session_id
)
os.makedirs(SESSION_VIOLATION_DIR, exist_ok=True)
st.session_state.session_violation_dir = SESSION_VIOLATION_DIR

# Session-local event queue (replaces the old process-global st._shared_event_queue)
if "_event_queue" not in st.session_state:
    st.session_state._event_queue = queue.Queue()
_event_queue = st.session_state._event_queue

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
_defaults = {
    "event_log": [],
    "last_status": "System idle. Ready for analysis.",
    "total_detections": 0,
    "helmet_count": 0,
    "violation_count": 0,
    "last_event_time": None,
    "unique_tracked_ids": set(),
    "frames_analyzed": 0,
    "class_counts": {"bicyclist": 0, "driver": 0, "helmet": 0, "no-helmet": 0},
    "confidence_scores": [],
    "recent_violations": [],
}
for _key, _val in _defaults.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
# Maximum number of confidence scores retained in session state.
# Prevents unbounded memory growth on long video sessions.
_MAX_CONF_SCORES = 10_000

def log_event(msg: str, is_helmet=None, class_name: str = None, conf: float = None):
    """Appends an event to the session state log and updates statistics."""
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    st.session_state.event_log.append(entry)
    st.session_state.last_status = msg
    st.session_state.last_event_time = timestamp

    if conf is not None:
        scores = st.session_state.confidence_scores
        if len(scores) < _MAX_CONF_SCORES:
            scores.append(float(conf))

    if class_name in st.session_state.class_counts:
        st.session_state.class_counts[class_name] += 1

    if is_helmet is True:
        st.session_state.total_detections += 1
        st.session_state.helmet_count += 1
    elif is_helmet is False:
        st.session_state.total_detections += 1
        st.session_state.violation_count += 1
    else:
        # bicyclist / driver detections (is_helmet=None) still count toward total
        st.session_state.total_detections += 1

def validate_image_file(file_obj) -> tuple[bool, str]:
    """Validates uploaded image file format and maximum size (15 MB)."""
    max_bytes = 15 * 1024 * 1024  # 15 MB
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    name = getattr(file_obj, "name", "sample.jpg")
    ext = os.path.splitext(name)[1].lower()

    if ext not in valid_exts:
        return False, f"Invalid format '{ext}'. Supported: JPG, JPEG, PNG, WEBP, BMP."

    size = getattr(file_obj, "size", 0)
    if size > max_bytes:
        return False, f"File size ({size / (1024*1024):.1f}MB) exceeds the 15MB maximum limit."
    return True, ""

def validate_video_file(file_obj) -> tuple[bool, str]:
    """Validates uploaded video file format and maximum size (100 MB)."""
    max_bytes = 100 * 1024 * 1024  # 100 MB
    valid_exts = {".mp4", ".avi", ".mov", ".mkv"}
    name = getattr(file_obj, "name", "sample.mp4")
    ext = os.path.splitext(name)[1].lower()

    if ext not in valid_exts:
        return False, f"Invalid format '{ext}'. Supported: MP4, AVI, MOV, MKV."

    size = getattr(file_obj, "size", 0)
    if size > max_bytes:
        return False, f"Video size ({size / (1024*1024):.1f}MB) exceeds the 100MB maximum limit."
    return True, ""

def is_point_in_roi(cx: float, cy: float, roi: dict, width: int, height: int) -> bool:
    """Evaluates whether the detection bounding box centroid falls within the ROI zone."""
    if not roi.get("enabled", False):
        return True
    rx1 = roi["x_min"] * width
    rx2 = roi["x_max"] * width
    ry1 = roi["y_min"] * height
    ry2 = roi["y_max"] * height
    return (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)

def draw_roi_overlay(img_bgr: np.ndarray, roi: dict) -> np.ndarray:
    """Draws a refined ROI zone boundary and label onto the image frame."""
    if not roi.get("enabled", False):
        return img_bgr
    h, w = img_bgr.shape[:2]
    rx1, ry1 = int(roi["x_min"] * w), int(roi["y_min"] * h)
    rx2, ry2 = int(roi["x_max"] * w), int(roi["y_max"] * h)

    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (170, 105, 23), -1)
    cv2.addWeighted(overlay, 0.12, img_bgr, 0.88, 0, img_bgr)
    cv2.rectangle(img_bgr, (rx1, ry1), (rx2, ry2), (170, 105, 23), 2)
    cv2.putText(
        img_bgr,
        f"ROI: {roi.get('preset_name', 'Zone')}",
        (rx1 + 8, max(ry1 + 24, 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (170, 105, 23),
        2
    )
    return img_bgr

# ==============================================================================
# SHELL & LAYOUT EXECUTION
# SESSION_VIOLATION_DIR is passed everywhere instead of a shared VIOLATION_DIR
# ==============================================================================
controls = render_sidebar_controls(model, SESSION_VIOLATION_DIR)
render_top_header()
render_live_detection_status(_event_queue, log_event)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# MAIN NAVIGATION TABS (5 MODES — ZERO EMOJIS)
# ==============================================================================
tab_image, tab_batch, tab_video, tab_cam, tab_analytics = st.tabs([
    "Single Image",
    "Batch Processing",
    "Video Analysis",
    "Live Camera",
    "Safety Analytics"
])

with tab_image:
    render_single_image_page(
        model=model,
        conf_thresh=controls["conf_thresh"],
        iou_thresh=controls["iou_thresh"],
        show_boxes=controls["show_boxes"],
        show_labels=controls["show_labels"],
        show_conf=controls["show_conf"],
        log_event_fn=log_event,
        validate_image_file_fn=validate_image_file,
        violation_dir=SESSION_VIOLATION_DIR,
        sample_image_path=SAMPLE_IMAGE_PATH
    )

with tab_batch:
    render_batch_page(
        model=model,
        conf_thresh=controls["conf_thresh"],
        iou_thresh=controls["iou_thresh"],
        show_boxes=controls["show_boxes"],
        show_labels=controls["show_labels"],
        show_conf=controls["show_conf"],
        log_event_fn=log_event,
        validate_image_file_fn=validate_image_file,
        sample_image_path=SAMPLE_IMAGE_PATH
    )

with tab_video:
    render_video_page(
        model=model,
        conf_thresh=controls["conf_thresh"],
        iou_thresh=controls["iou_thresh"],
        show_boxes=controls["show_boxes"],
        show_labels=controls["show_labels"],
        show_conf=controls["show_conf"],
        enable_tracking=controls["enable_tracking"],
        video_stride=controls["video_stride"],
        log_event_fn=log_event,
        validate_video_file_fn=validate_video_file,
        is_point_in_roi_fn=is_point_in_roi,
        draw_roi_overlay_fn=draw_roi_overlay,
        violation_dir=SESSION_VIOLATION_DIR,
        sample_video_path=SAMPLE_VIDEO_PATH
    )

with tab_cam:
    render_live_camera_page(
        model=model,
        conf_thresh=controls["conf_thresh"],
        iou_thresh=controls["iou_thresh"],
        show_boxes=controls["show_boxes"],
        show_labels=controls["show_labels"],
        show_conf=controls["show_conf"],
        event_queue=_event_queue,
        violation_dir=SESSION_VIOLATION_DIR
    )

with tab_analytics:
    render_analytics_page(
        violation_dir=SESSION_VIOLATION_DIR
    )

# ==============================================================================
# FOOTER
# ==============================================================================
render_footer()
