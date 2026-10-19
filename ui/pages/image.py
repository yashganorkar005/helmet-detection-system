"""
Single Image Inspection page module for Helmet Detection & Road Safety Analyzer.
Performs YOLOv8 inference on single uploaded images or sample demonstration media.
"""

import os
import time
import io
import json
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import streamlit as st
from ui.annotations import plot_with_priority_labels

def render_single_image_page(
    model,
    conf_thresh: float,
    iou_thresh: float,
    show_boxes: bool,
    show_labels: bool,
    show_conf: bool,
    log_event_fn,
    validate_image_file_fn,
    violation_dir: str,
    sample_image_path: str
):
    """Renders the single image detection and analysis view."""
    st.markdown(
        """
        <p style="color: var(--text-secondary); font-size: 0.86rem; margin-bottom: 1.25rem;">
            Inspect individual roadway snapshots or test with sample traffic imagery.
        </p>
        """,
        unsafe_allow_html=True
    )

    col_upload, col_demo = st.columns([3, 1], vertical_alignment="bottom")
    with col_upload:
        uploaded_img = st.file_uploader(
            "Select roadway snapshot (Max 15MB)",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="single_img_uploader"
        )
    with col_demo:
        demo_btn = st.button("Load Demo Image", use_container_width=True)

    img_to_process = None
    source_label = ""

    if demo_btn and os.path.exists(sample_image_path):
        img_to_process = Image.open(sample_image_path).convert("RGB")
        source_label = "Demo Sample (Traffic Intersection)"
    elif uploaded_img is not None:
        valid, err_msg = validate_image_file_fn(uploaded_img)
        if not valid:
            st.error(err_msg)
        else:
            img_to_process = Image.open(uploaded_img).convert("RGB")
            source_label = uploaded_img.name

    if img_to_process is not None:
        orig_np = np.array(img_to_process)
        h, w = orig_np.shape[:2]

        t0 = time.time()
        results = model.predict(
            orig_np,
            conf=conf_thresh,
            iou=iou_thresh,
            verbose=False
        )
        latency_ms = (time.time() - t0) * 1000

        annotated_np = plot_with_priority_labels(
            results[0],
            show_boxes=show_boxes,
            show_labels=show_labels,
            show_conf=show_conf
        )

        st.session_state.frames_analyzed += 1

        # Side-by-side comparison
        col_orig, col_annot = st.columns(2)
        with col_orig:
            st.markdown(f"**Original Image** · `{w}×{h}` px")
            st.image(orig_np, use_container_width=True)
        with col_annot:
            st.markdown(f"**Annotated Detection** · `{latency_ms:.1f} ms` latency")
            st.image(annotated_np, use_container_width=True)

        # Parse detections
        detections_data = []
        boxes = results[0].boxes
        img_violations = 0
        img_helmets = 0

        for box in boxes:
            cls_id = int(box.cls[0])
            c_name = model.names[cls_id].lower()
            conf_val = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            is_violation = (c_name == "no-helmet")
            if is_violation:
                img_violations += 1
                log_event_fn(f"Violation in {source_label}: No-Helmet (conf {conf_val:.2f})", is_helmet=False, class_name="no-helmet", conf=conf_val)
            elif c_name == "helmet":
                img_helmets += 1
                log_event_fn(f"Helmet in {source_label} (conf {conf_val:.2f})", is_helmet=True, class_name="helmet", conf=conf_val)
            else:
                log_event_fn(f"Detected {c_name} in {source_label} (conf {conf_val:.2f})", is_helmet=None, class_name=c_name, conf=conf_val)

            detections_data.append({
                "Class": c_name.capitalize(),
                "Confidence": f"{conf_val * 100:.1f}%",
                "Category": "Violation" if is_violation else ("Compliant" if c_name == "helmet" else "Road User"),
                "Box (x1, y1, x2, y2)": f"[{int(xyxy[0])}, {int(xyxy[1])}, {int(xyxy[2])}, {int(xyxy[3])}]"
            })

        # Save violation snapshot if violations found
        if img_violations > 0:
            snapshot_name = f"violation_img_{int(time.time())}.jpg"
            v_dir = getattr(st.session_state, "session_violation_dir", violation_dir)
            save_path = os.path.join(v_dir, snapshot_name)
            cv2.imwrite(save_path, cv2.cvtColor(annotated_np, cv2.COLOR_RGB2BGR))
            st.session_state.recent_violations.append({
                "path": save_path,
                "name": snapshot_name,
                "time": time.strftime("%H:%M:%S"),
                "violations": img_violations
            })

        # Results table & downloads
        if detections_data:
            df_det = pd.DataFrame(detections_data)
            st.markdown("##### Detections Breakdown")
            st.dataframe(df_det, use_container_width=True)

            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                buf_img = io.BytesIO()
                Image.fromarray(annotated_np).save(buf_img, format="JPEG")
                st.download_button(
                    "Download Annotated Image",
                    data=buf_img.getvalue(),
                    file_name=f"detected_{int(time.time())}.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
            with col_d2:
                csv_bytes = df_det.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Export Results (CSV)",
                    data=csv_bytes,
                    file_name=f"detections_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col_d3:
                json_bytes = json.dumps(detections_data, indent=2).encode("utf-8")
                st.download_button(
                    "Export Results (JSON)",
                    data=json_bytes,
                    file_name=f"detections_{int(time.time())}.json",
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.info("No road users or helmets identified above the current confidence threshold.")
