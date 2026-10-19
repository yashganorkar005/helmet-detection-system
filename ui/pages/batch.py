"""
Batch Image Processing page module for Helmet Detection & Road Safety Analyzer.
Executes batch inference across multiple uploaded roadway images and creates in-memory ZIP archives.
"""

import os
import time
import io
import json
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ui.annotations import plot_with_priority_labels

def render_batch_page(
    model,
    conf_thresh: float,
    iou_thresh: float,
    show_boxes: bool,
    show_labels: bool,
    show_conf: bool,
    log_event_fn,
    validate_image_file_fn,
    sample_image_path: str
):
    """Renders the batch image processing view."""
    st.markdown(
        """
        <p style="color: var(--text-secondary); font-size: 0.86rem; margin-bottom: 1.25rem;">
            Process multiple roadway images in a single batch. Download individual annotated images or a consolidated ZIP archive.
        </p>
        """,
        unsafe_allow_html=True
    )

    col_batch_up, col_batch_demo = st.columns([3, 1], vertical_alignment="bottom")
    with col_batch_up:
        batch_files = st.file_uploader(
            "Select multiple images for batch inspection",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=True,
            key="batch_img_uploader"
        )
    with col_batch_demo:
        demo_batch_btn = st.button("Test Demo Batch", use_container_width=True)

    files_to_process = []
    if demo_batch_btn and os.path.exists(sample_image_path):
        files_to_process = [
            ("sample_street_view1.jpg", sample_image_path),
            ("sample_street_view2.jpg", sample_image_path)
        ]
    elif batch_files:
        files_to_process = [(f.name, f) for f in batch_files]

    if files_to_process:
        st.markdown(f"**Found {len(files_to_process)} image(s) in batch queue.** Click below to start execution:")
        if st.button("Process Image Batch", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            batch_summary = []
            annotated_images_map = {}
            total_b_violations = 0
            total_b_helmets = 0

            for idx, (fname, fitem) in enumerate(files_to_process):
                status_text.text(f"Processing image {idx+1}/{len(files_to_process)}: {fname}...")

                if isinstance(fitem, str):
                    img_np = np.array(Image.open(fitem).convert("RGB"))
                else:
                    valid, _ = validate_image_file_fn(fitem)
                    if not valid:
                        continue
                    img_np = np.array(Image.open(fitem).convert("RGB"))

                t0 = time.time()
                res = model.predict(img_np, conf=conf_thresh, iou=iou_thresh, verbose=False)
                dur = (time.time() - t0) * 1000

                ann_np = plot_with_priority_labels(res[0], show_boxes=show_boxes, show_labels=show_labels, show_conf=show_conf)
                annotated_images_map[fname] = ann_np

                n_helmets = sum(1 for b in res[0].boxes if model.names[int(b.cls[0])].lower() == "helmet")
                n_violations = sum(1 for b in res[0].boxes if model.names[int(b.cls[0])].lower() == "no-helmet")
                n_drivers = sum(1 for b in res[0].boxes if model.names[int(b.cls[0])].lower() == "driver")
                n_cyclists = sum(1 for b in res[0].boxes if model.names[int(b.cls[0])].lower() == "bicyclist")

                total_b_helmets += n_helmets
                total_b_violations += n_violations
                st.session_state.frames_analyzed += 1

                for b in res[0].boxes:
                    cname = model.names[int(b.cls[0])].lower()
                    cval = float(b.conf[0])
                    if cname == "helmet":
                        log_event_fn(f"Batch {fname}: Helmet detected", is_helmet=True, class_name=cname, conf=cval)
                    elif cname == "no-helmet":
                        log_event_fn(f"Batch {fname}: Violation detected", is_helmet=False, class_name=cname, conf=cval)
                    else:
                        log_event_fn(f"Batch {fname}: {cname} detected", is_helmet=None, class_name=cname, conf=cval)

                batch_summary.append({
                    "Filename": fname,
                    "Total Detections": len(res[0].boxes),
                    "Helmets": n_helmets,
                    "Violations": n_violations,
                    "Drivers": n_drivers,
                    "Bicyclists": n_cyclists,
                    "Inference (ms)": round(dur, 1)
                })

                progress_bar.progress((idx + 1) / len(files_to_process))

            status_text.success(f"Batch completed: Processed {len(files_to_process)} images successfully.")

            # Batch Summary Metrics
            df_batch = pd.DataFrame(batch_summary)
            st.markdown("##### Batch Processing Results Summary")
            st.dataframe(df_batch, use_container_width=True)

            # In-Memory ZIP Archive Creation
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                csv_data = df_batch.to_csv(index=False)
                zip_file.writestr("batch_summary.csv", csv_data)

                json_data = json.dumps(batch_summary, indent=2)
                zip_file.writestr("batch_summary.json", json_data)

                for im_name, im_arr in annotated_images_map.items():
                    im_buf = io.BytesIO()
                    Image.fromarray(im_arr).save(im_buf, format="JPEG")
                    base_name = os.path.splitext(im_name)[0]
                    zip_file.writestr(f"annotated_{base_name}.jpg", im_buf.getvalue())

            zip_buffer.seek(0)
            st.download_button(
                "Download Complete Batch Archive (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"batch_inspection_{int(time.time())}.zip",
                mime="application/zip",
                use_container_width=True
            )

            # Visual Gallery of Processed Images
            st.markdown("##### Annotated Image Gallery")
            gallery_cols = st.columns(min(3, max(1, len(annotated_images_map))))
            for i, (im_name, im_arr) in enumerate(annotated_images_map.items()):
                col = gallery_cols[i % len(gallery_cols)]
                with col:
                    st.image(im_arr, caption=im_name, use_container_width=True)
