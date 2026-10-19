"""
Live Camera WebRTC stream page module for Helmet Detection & Road Safety Analyzer.
Asynchronously processes edge camera frames with live bounding boxes, queue event signaling, and snapshot capture.
"""

import os
import time
import cv2
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ui.annotations import plot_with_priority_labels

def create_video_processor_class(model, conf_thresh, iou_thresh, show_boxes, show_labels, show_conf, event_queue, violation_dir):
    """Factory to create a VideoProcessor class bound with active threshold parameters."""
    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.last_save = 0
            self.event_queue = event_queue
            self.model = model
            self.conf = conf_thresh
            self.iou = iou_thresh
            self.show_b = show_boxes
            self.show_l = show_labels
            self.show_c = show_conf
            self.v_dir = violation_dir

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = self.model.predict(
                img,
                conf=self.conf,
                iou=self.iou,
                verbose=False
            )
            annotated = plot_with_priority_labels(
                results[0],
                show_boxes=self.show_b,
                show_labels=self.show_l,
                show_conf=self.show_c
            )

            for box in results[0].boxes:
                cname = self.model.names[int(box.cls[0])].lower()
                self.event_queue.put(cname)

                if cname == "no-helmet" and (time.time() - self.last_save > 4.0):
                    self.last_save = time.time()
                    snap_path = os.path.join(self.v_dir, f"cam_violation_{int(time.time())}.jpg")
                    cv2.imwrite(snap_path, annotated)

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    return VideoProcessor

def render_live_camera_page(
    model,
    conf_thresh: float,
    iou_thresh: float,
    show_boxes: bool,
    show_labels: bool,
    show_conf: bool,
    event_queue,
    violation_dir: str
):
    """Renders the live camera streaming interface."""
    st.markdown(
        """
        <p style="color: var(--text-secondary); font-size: 0.86rem; margin-bottom: 1.25rem;">
            Real-time edge camera monitoring via WebRTC. Click <strong>START</strong> below and allow browser camera access.
        </p>
        """,
        unsafe_allow_html=True
    )

    col_cam, col_info = st.columns([3, 1])

    processor_cls = create_video_processor_class(
        model, conf_thresh, iou_thresh, show_boxes, show_labels, show_conf, event_queue, violation_dir
    )

    with col_cam:
        webrtc_streamer(
            key="helmet-cam-stream",
            video_processor_factory=processor_cls,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

    with col_info:
        st.markdown(
            f"""
            <div style="background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:1.25rem; box-shadow:var(--shadow-sm);">
                <div style="font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.8px; color:var(--text-secondary); margin-bottom:0.85rem;">
                    Live Stream Telemetry
                </div>
                <div style="font-size:0.82rem; color:var(--text-secondary); line-height:2.1;">
                    <div>Stream: <strong>WebRTC Low-Latency</strong></div>
                    <div>Model: <strong>YOLOv8 PyTorch</strong></div>
                    <div>Conf Threshold: <strong>{conf_thresh:.2f}</strong></div>
                    <div>Violation Interval: <strong>4.0s</strong></div>
                    <div>Tracking: <strong>Active</strong></div>
                    <div style="margin-top:0.75rem; padding-top:0.75rem; border-top:1px solid var(--border);">
                        Session Violations: <strong style="color:var(--violation-red);">{st.session_state.violation_count}</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        if st.button("Reset Live Cam Counters", use_container_width=True):
            st.session_state.helmet_count = 0
            st.session_state.violation_count = 0
            st.session_state.total_detections = 0
            st.rerun()

    with st.expander("Webcam Troubleshooting & Permissions Guide"):
        st.markdown("""
        - **Camera Permission Denied:** Click the lock or camera icon on the left side of your browser's URL address bar, change camera permissions to **Allow**, and refresh the page.
        - **Camera in Use:** Ensure other programs (Zoom, Teams, Skype, or other browser tabs) are not holding exclusive control of your webcam.
        - **HTTPS / Localhost Requirement:** WebRTC video streaming requires either a `localhost` URL or an `https://` secure connection.
        """)
