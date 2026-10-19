"""
Layout and shell components for Helmet Detection & Road Safety Analyzer.
Contains app header, sidebar controls, live detection status fragment, and footer.
Completely free of emojis; styled using Apple-inspired light neutral tokens.
"""

import os
import time
import queue
import streamlit as st
from ui.theme import CANVAS, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, BORDER
from ui.icons import (
    icon_shield, icon_settings, icon_refresh, icon_trash,
    icon_info, icon_check, icon_alert, icon_chart
)
from ui.components import (
    render_page_header, render_stat_card, render_section_header,
    render_callout, render_badge
)

def get_folder_info(folder_path: str) -> tuple[int, str]:
    """Returns file count and human-readable storage size of a directory."""
    if not os.path.exists(folder_path):
        return 0, "0 KB"
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    total_bytes = sum(os.path.getsize(f) for f in files)
    if total_bytes < 1024:
        size_str = f"{total_bytes} B"
    elif total_bytes < 1024 * 1024:
        size_str = f"{total_bytes / 1024:.1f} KB"
    else:
        size_str = f"{total_bytes / (1024 * 1024):.2f} MB"
    return len(files), size_str

def render_top_header():
    """Renders top hero branding and informational expanders without emojis."""
    render_page_header(
        title="Helmet Detection & Road Safety Analyzer",
        subtitle="Deep Learning Computer Vision Platform · YOLOv8 & ByteTrack Multi-Object Intelligence",
        badge_text="System Online · Local Inference",
        badge_active=True
    )

    with st.expander("Platform Architecture, Capabilities & Trained Classes", expanded=False):
        st.markdown("""
        This platform delivers real-time and post-capture road safety compliance monitoring powered by a custom **YOLOv8** deep learning architecture and **ByteTrack** multi-object tracking.

        **Trained Classes & Visual Identifiers:**
        - **Helmet (Class ID 2):** Motorcyclist or cyclist wearing protective headwear. Flagged as **Compliant** (Green).
        - **No-Helmet (Class ID 3):** Rider operating without certified head protection. Flagged as **Violation** (Crimson).
        - **Driver (Class ID 1):** Operator of a motorcycle, scooter, or moped. Flagged as **Operator** (Slate).
        - **Bicyclist (Class ID 0):** Bicycle rider on the roadway. Flagged as **Cyclist** (Amber).
        """)

    with st.expander("Privacy Policy, Local Execution Guarantee & Data Handling", expanded=False):
        st.markdown("""
        - **100% Local Inference:** All neural network inference, frame decoding, and tracking run directly on the host CPU.
        - **Zero Cloud Data Transmission:** No captured images, video streams, or telemetry leave your local environment.
        - **No User Tracking or Accounts:** Access is direct and unauthenticated. No tracking cookies or advertising telemetry.
        - **Volatile Storage & User Control:** Analysis artifacts are automatically purged from memory. You can wipe all saved violation records at any time via the Control Panel.
        """)

    with st.expander("Operational Scope & Research Disclaimer", expanded=False):
        st.markdown("""
        This software is engineered strictly for **academic research, smart city safety analytics, and computer vision demonstrations**. It does not constitute an automated legal enforcement or ticketing system. Environmental variables including low-light conditions, extreme angles, occlusions, or hooded apparel may impact detection confidence.
        """)

def render_sidebar_controls(model, violation_dir: str) -> dict:
    """Renders the sidebar control panel with model parameters and storage actions."""
    with st.sidebar:
        header_sb = (
            f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:1.25rem;">'
            f'<div style="color:var(--primary-accent);">{icon_settings(size=22)}</div>'
            f'<div>'
            f'<div class="sidebar-title">Control Panel</div>'
            f'<div class="sidebar-subtitle">AI Detection Config</div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(header_sb, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section-heading">Detection Thresholds</div>', unsafe_allow_html=True)
        conf_thresh = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.35,
            step=0.05,
            help="Minimum confidence score required for an object detection to be accepted."
        )
        iou_thresh = st.slider(
            "IOU / NMS Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.45,
            step=0.05,
            help="Intersection over Union threshold for Non-Maximum Suppression to filter duplicate boxes."
        )

        st.markdown('<div class="sidebar-section-heading">Visual Overlays</div>', unsafe_allow_html=True)
        show_boxes = st.checkbox("Show Bounding Boxes", value=True)
        show_labels = st.checkbox("Show Class Labels", value=True)
        show_conf = st.checkbox("Show Confidence Score", value=True)

        st.markdown('<div class="sidebar-section-heading">Tracking &amp; Hardware</div>', unsafe_allow_html=True)
        device_opt = st.selectbox(
            "Inference Device",
            ["Auto (CPU)", "CPU"],
            index=0,
            help="PyTorch optimized CPU inference host."
        )
        enable_tracking = st.checkbox(
            "Multi-Object Tracking (ByteTrack)",
            value=True,
            help="Assigns persistent IDs to riders across video frames to prevent duplicate violation counts."
        )
        video_stride = st.slider(
            "Video Frame Stride",
            min_value=1,
            max_value=5,
            value=2,
            help="Process every Nth frame. Higher stride increases throughput speed on CPU."
        )

        st.markdown('<div class="sidebar-section-heading">Storage & Session Data</div>', unsafe_allow_html=True)
        v_count, v_size = get_folder_info(violation_dir)
        storage_box = (
            f'<div style="background:var(--surface-subtle); border:1px solid var(--border); border-radius:8px; padding:10px 12px; margin-bottom:12px; font-size:0.78rem;">'
            f'<div style="display:flex; justify-content:space-between; margin-bottom:4px;">'
            f'<span style="color:var(--text-secondary);">Saved Violations:</span>'
            f'<strong style="color:var(--violation-red);">{v_count}</strong>'
            f'</div>'
            f'<div style="display:flex; justify-content:space-between;">'
            f'<span style="color:var(--text-secondary);">Storage Allocated:</span>'
            f'<strong style="color:var(--text-primary);">{v_size}</strong>'
            f'</div>'
            f'</div>'
        )
        st.markdown(storage_box, unsafe_allow_html=True)

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Reset Stats", use_container_width=True, help="Reset in-memory counters", key="reset_stats_btn", type="secondary"):
                st.session_state.total_detections = 0
                st.session_state.helmet_count = 0
                st.session_state.violation_count = 0
                st.session_state.unique_tracked_ids = set()
                st.session_state.frames_analyzed = 0
                st.session_state.class_counts = {"bicyclist": 0, "driver": 0, "helmet": 0, "no-helmet": 0}
                st.session_state.confidence_scores = []
                st.session_state.event_log = []
                st.session_state.last_status = "Statistics reset by user."
                st.rerun()

        with col_btn2:
            if st.button("Delete Files", use_container_width=True, help="Permanently delete violation images", key="delete_files_btn", type="secondary"):
                for fname in os.listdir(violation_dir):
                    fpath = os.path.join(violation_dir, fname)
                    try:
                        if os.path.isfile(fpath):
                            os.unlink(fpath)
                    except Exception:
                        pass
                st.session_state.recent_violations = []
                st.rerun()

    return {
        "conf_thresh": conf_thresh,
        "iou_thresh": iou_thresh,
        "show_boxes": show_boxes,
        "show_labels": show_labels,
        "show_conf": show_conf,
        "enable_tracking": enable_tracking,
        "video_stride": video_stride
    }

def render_live_detection_status(event_queue, log_event_fn):
    """
    Renders the live detection metrics fragment.
    Updates every 2 seconds without full-page reloads.
    Features a clean Reset button to zero out metrics and drain the event queue.
    """
    @st.fragment(run_every="2s")
    def _fragment():
        # Drain WebRTC event queue
        while not event_queue.empty():
            try:
                label = event_queue.get_nowait()
                if label.strip().lower() == "helmet":
                    log_event_fn("Helmet detected (Live Camera)", is_helmet=True, class_name="helmet")
                elif label.strip().lower() == "no-helmet":
                    log_event_fn("Violation: No Helmet detected (Live Camera)", is_helmet=False, class_name="no-helmet")
            except queue.Empty:
                break

        total = st.session_state.total_detections
        helmets = st.session_state.helmet_count
        violations = st.session_state.violation_count
        compliance = f"{(helmets / total * 100):.1f}%" if total > 0 else "N/A"
        last_time = st.session_state.last_event_time or "None"

        # Header Row with Reset Button (No emojis)
        col_lbl, col_rst = st.columns([5, 1], vertical_alignment="center")
        with col_lbl:
            render_section_header("Live Detection Status")
        with col_rst:
            if st.button("Reset", key="reset_live_status_btn", use_container_width=True, help="Reset Live Detection Status values (Total Detections, Helmets, Violations, Compliance Rate) to zero", type="secondary"):
                st.session_state.total_detections = 0
                st.session_state.helmet_count = 0
                st.session_state.violation_count = 0
                st.session_state.last_status = "Live detection status reset to zero."
                st.session_state.last_event_time = None
                while not event_queue.empty():
                    try:
                        event_queue.get_nowait()
                    except queue.Empty:
                        break
                st.rerun()

        # 4 Stat Cards
        c1 = render_stat_card("Total Detections", total, "Raw bounding boxes", tone="primary", icon_svg=icon_chart(size=18, color="#1769AA"))
        c2 = render_stat_card("Helmets Detected", helmets, "Compliant riders", tone="compliant", icon_svg=icon_check(size=18, color="#237A57"))
        c3 = render_stat_card("Violations", violations, "Non-compliant events", tone="violation", icon_svg=icon_alert(size=18, color="#B42318"))
        c4 = render_stat_card("Compliance Rate", compliance, "Helmet / Total Ratio", tone="warning", icon_svg=icon_shield(size=18, color="#A66308"))

        st.markdown(f'<div class="stat-grid">{c1}{c2}{c3}{c4}</div>', unsafe_allow_html=True)

        # Status & Activity Log Rows
        col_stat, col_log = st.columns([1, 2])
        with col_stat:
            status = st.session_state.last_status
            if "No Helmet" in status or "Violation" in status:
                css_class = "danger"
            elif "detected" in status.lower() and "no" not in status.lower():
                css_class = "safe"
            else:
                css_class = "idle"

            status_html = (
                f'<div class="status-indicator-box">'
                f'<span class="stat-label">System Status</span>'
                f'<div class="status-pill {css_class}">'
                f'<div class="status-pulse-dot"></div>'
                f'<div>{status}</div>'
                f'</div>'
                f'<div class="status-timestamp">Last Event: {last_time}</div>'
                f'</div>'
            )
            st.markdown(status_html, unsafe_allow_html=True)

        with col_log:
            header_html = (
                '<div class="terminal-wrapper">'
                '<div class="terminal-header">'
                '<span class="terminal-title">System Activity Log</span>'
                '<span style="font-size:0.7rem; color:var(--text-muted);">stdout</span>'
                '</div>'
            )
            st.markdown(header_html, unsafe_allow_html=True)
            if st.session_state.event_log:
                st.text_area(
                    label="",
                    value="\n".join(st.session_state.event_log[-12:]),
                    height=140,
                    label_visibility="collapsed"
                )
            else:
                st.markdown(
                    '<div style="padding:16px 18px; font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:0.78rem; color:var(--text-muted);">Ready. Waiting for inspection data...</div>',
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)

    _fragment()

def render_footer():
    """Renders the clean minimal footer bar without emojis."""
    st.markdown(
        """
        <div class="footer-bar">
            <p>Helmet Detection &amp; Road Safety Analyzer · Professional AI Safety System</p>
            <p>Powered by Ultralytics YOLOv8 · ByteTrack Multi-Object Intelligence · Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )
