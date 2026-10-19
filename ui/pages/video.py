"""
Video Analysis page module for Helmet Detection & Road Safety Analyzer.
Provides ByteTrack multi-object tracking, ROI lane filtering, timeline analytics, and violation extraction.
"""

import os
import time
import json
import tempfile
import cv2
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from ui.theme import get_theme_tokens
from ui.annotations import plot_with_priority_labels

def render_video_page(
    model,
    conf_thresh: float,
    iou_thresh: float,
    show_boxes: bool,
    show_labels: bool,
    show_conf: bool,
    enable_tracking: bool,
    video_stride: int,
    log_event_fn,
    validate_video_file_fn,
    is_point_in_roi_fn,
    draw_roi_overlay_fn,
    violation_dir: str,
    sample_video_path: str
):
    """Renders the video inference, tracking, and ROI analysis view."""
    st.markdown(
        """
        <p style="color: var(--text-secondary); font-size: 0.86rem; margin-bottom: 1.25rem;">
            Execute video analysis with ByteTrack multi-object tracking and optional Region-of-Interest (ROI) lane filtering.
        </p>
        """,
        unsafe_allow_html=True
    )

    with st.expander("Region of Interest (ROI) & Lane Filtering Options", expanded=False):
        col_roi1, col_roi2 = st.columns(2)
        with col_roi1:
            roi_enabled = st.checkbox("Enable Region of Interest (ROI) Filtering", value=False)
            roi_preset = st.selectbox(
                "ROI Lane Preset",
                ["Full Frame", "Left Lane (0% - 50% X)", "Right Lane (50% - 100% X)", "Center Lane (25% - 75% X)", "Custom Zone"]
            )

        roi_config = {"enabled": roi_enabled, "preset_name": roi_preset}
        if roi_preset == "Full Frame":
            roi_config.update({"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0})
        elif roi_preset == "Left Lane (0% - 50% X)":
            roi_config.update({"x_min": 0.0, "x_max": 0.50, "y_min": 0.0, "y_max": 1.0})
        elif roi_preset == "Right Lane (50% - 100% X)":
            roi_config.update({"x_min": 0.50, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0})
        elif roi_preset == "Center Lane (25% - 75% X)":
            roi_config.update({"x_min": 0.25, "x_max": 0.75, "y_min": 0.0, "y_max": 1.0})
        else:
            with col_roi2:
                x_min = st.slider("X Min Ratio", 0.0, 0.9, 0.1, 0.05)
                x_max = st.slider("X Max Ratio", 0.1, 1.0, 0.9, 0.05)
                y_min = st.slider("Y Min Ratio", 0.0, 0.9, 0.0, 0.05)
                y_max = st.slider("Y Max Ratio", 0.1, 1.0, 1.0, 0.05)
                roi_config.update({"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max})

    col_v_up, col_v_demo = st.columns([3, 1], vertical_alignment="bottom")
    with col_v_up:
        uploaded_video = st.file_uploader(
            "Upload roadway video footage (MP4, AVI, MOV, MKV · Max 100MB)",
            type=["mp4", "avi", "mov", "mkv"],
            key="video_uploader"
        )
    with col_v_demo:
        demo_video_btn = st.button("Load Demo Video", use_container_width=True)

    active_video_path = None
    is_temp_file = False

    if demo_video_btn and os.path.exists(sample_video_path):
        active_video_path = sample_video_path
    elif uploaded_video is not None:
        valid, err_msg = validate_video_file_fn(uploaded_video)
        if not valid:
            st.error(err_msg)
        else:
            # Write the upload to a session-specific temp file.
            # Store the path in session_state so any subsequent rerun triggered
            # by a second upload can delete the previous orphaned temp file first.
            v_dir = getattr(st.session_state, "session_violation_dir", violation_dir)
            prev_temp = st.session_state.get("_video_temp_path")
            if prev_temp and os.path.exists(prev_temp):
                try:
                    os.unlink(prev_temp)
                except Exception:
                    pass
            temp_vid = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4",
                dir=v_dir
            )
            temp_vid.write(uploaded_video.read())
            temp_vid.close()
            active_video_path = temp_vid.name
            st.session_state["_video_temp_path"] = active_video_path
            is_temp_file = True

    if active_video_path is not None:
        # Open a probe cap just to read metadata for the telemetry banner.
        # The analysis cap is opened separately inside the button block so that
        # the probe cap is always released even when the button isn't pressed.
        probe_cap = cv2.VideoCapture(active_video_path)
        total_frames = int(probe_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = probe_cap.get(cv2.CAP_PROP_FPS) or 25.0
        v_width = int(probe_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_height = int(probe_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps if fps > 0 else 0
        probe_cap.release()

        st.markdown(
            f"""
            <div style="background:var(--surface-subtle); border:1px solid var(--border); border-radius:8px; padding:10px 14px; margin:10px 0; font-size:0.82rem;">
                <span style="color:var(--text-secondary); font-weight:600;">Video Telemetry:</span>
                &nbsp; Resolution: <code>{v_width}×{v_height}</code> &nbsp;·&nbsp;
                FPS: <code>{fps:.1f}</code> &nbsp;·&nbsp;
                Frames: <code>{total_frames}</code> &nbsp;·&nbsp;
                Duration: <code>{duration_sec:.1f}s</code>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("Start Video Analysis", type="primary"):
            progress_bar = st.progress(0)
            frame_view = st.empty()
            stat_placeholder = st.empty()

            timeline_records = []
            violation_events = []
            video_tracked_ids = set()
            frame_idx = 0
            processed_count = 0
            v_helmets = 0
            v_violations = 0

            start_proc_time = time.time()
            cap = cv2.VideoCapture(active_video_path)

            try:
                while cap.isOpened():
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break

                    frame_idx += 1
                    if frame_idx % video_stride != 0:
                        continue

                    processed_count += 1
                    st.session_state.frames_analyzed += 1
                    current_time_sec = frame_idx / fps

                    # Model Inference / Tracking
                    if enable_tracking:
                        results = model.track(
                            frame_bgr,
                            persist=True,
                            tracker="bytetrack.yaml",
                            conf=conf_thresh,
                            iou=iou_thresh,
                            verbose=False
                        )
                    else:
                        results = model.predict(
                            frame_bgr,
                            conf=conf_thresh,
                            iou=iou_thresh,
                            verbose=False
                        )

                    annotated_bgr = plot_with_priority_labels(
                        results[0],
                        show_boxes=show_boxes,
                        show_labels=show_labels,
                        show_conf=show_conf
                    )

                    # Draw ROI overlay if enabled
                    if roi_config.get("enabled", False):
                        annotated_bgr = draw_roi_overlay_fn(annotated_bgr, roi_config)

                    frame_h, frame_w = frame_bgr.shape[:2]
                    boxes = results[0].boxes
                    frame_helmets = 0
                    frame_violations = 0

                    for box in boxes:
                        xyxy = box.xyxy[0].tolist()
                        cx = (xyxy[0] + xyxy[2]) / 2.0
                        cy = (xyxy[1] + xyxy[3]) / 2.0

                        if not is_point_in_roi_fn(cx, cy, roi_config, frame_w, frame_h):
                            continue

                        cls_id = int(box.cls[0])
                        cname = model.names[cls_id].lower()
                        conf_val = float(box.conf[0])

                        track_id = int(box.id[0]) if (box.id is not None) else None
                        if track_id is not None:
                            video_tracked_ids.add(track_id)
                            st.session_state.unique_tracked_ids.add(track_id)

                        if cname == "helmet":
                            frame_helmets += 1
                            v_helmets += 1
                            log_event_fn(f"Helmet detected (Video @ {current_time_sec:.1f}s)", is_helmet=True, class_name="helmet", conf=conf_val)
                        elif cname == "no-helmet":
                            frame_violations += 1
                            v_violations += 1
                            log_event_fn(f"Violation: No Helmet (Video @ {current_time_sec:.1f}s)", is_helmet=False, class_name="no-helmet", conf=conf_val)

                            # Throttle violation snapshots to max 1 per second
                            if len(violation_events) == 0 or (current_time_sec - violation_events[-1]["time_sec"]) > 1.0:
                                v_snap_name = f"vid_violation_{int(time.time())}_{frame_idx}.jpg"
                                v_dir = getattr(st.session_state, "session_violation_dir", violation_dir)
                                v_snap_path = os.path.join(v_dir, v_snap_name)
                                cv2.imwrite(v_snap_path, annotated_bgr)
                                violation_events.append({
                                    "time_sec": round(current_time_sec, 2),
                                    "timestamp": f"{int(current_time_sec // 60):02d}:{current_time_sec % 60:04.1f}",
                                    "track_id": track_id or "None",
                                    "conf": f"{conf_val * 100:.1f}%",
                                    "path": v_snap_path
                                })
                        else:
                            log_event_fn(f"{cname} detected (Video @ {current_time_sec:.1f}s)", is_helmet=None, class_name=cname, conf=conf_val)

                    timeline_records.append({
                        "Time (s)": round(current_time_sec, 1),
                        "Helmets": frame_helmets,
                        "Violations": frame_violations,
                        "Total Active": len(boxes)
                    })

                    # Update preview
                    frame_view.image(annotated_bgr, channels="BGR", use_container_width=True)
                    progress_pct = min(1.0, frame_idx / total_frames) if total_frames > 0 else 0
                    progress_bar.progress(progress_pct)

                    elapsed = time.time() - start_proc_time
                    fps_rate = processed_count / elapsed if elapsed > 0 else 0
                    stat_placeholder.markdown(
                        f"**Processing Frame:** `{frame_idx}/{total_frames}` &nbsp;·&nbsp; "
                        f"**Throughput:** `{fps_rate:.1f} FPS` &nbsp;·&nbsp; "
                        f"**Violations Flagged:** <span style='color:var(--violation-red); font-weight:600;'>{v_violations}</span>",
                        unsafe_allow_html=True
                    )

            finally:
                cap.release()
                if is_temp_file and active_video_path and os.path.exists(active_video_path):
                    try:
                        os.unlink(active_video_path)
                    except Exception:
                        pass
                # Clear the session-state pointer so a next rerun doesn't
                # try to delete a path that no longer exists.
                st.session_state.pop("_video_temp_path", None)

            st.success(f"Video analysis complete: Processed {processed_count} frames successfully.")

            # Summary Metrics Row
            col_vm1, col_vm2, col_vm3, col_vm4 = st.columns(4)
            with col_vm1:
                st.metric("Total Frames Analyzed", processed_count)
            with col_vm2:
                st.metric("Unique Tracked Entities", len(video_tracked_ids) if video_tracked_ids else "N/A (Tracking off)")
            with col_vm3:
                st.metric("Total Violations Flagged", v_violations)
            with col_vm4:
                v_total = v_helmets + v_violations
                comp_pct = f"{(v_helmets / v_total * 100):.1f}%" if v_total > 0 else "N/A"
                st.metric("Safety Compliance Rate", comp_pct)

            # Timeline Chart (Altair)
            if timeline_records:
                st.markdown("##### Detection Timeline Chart")
                df_time = pd.DataFrame(timeline_records)

                # Retrieve active theme tokens for dynamic chart styling
                theme_mode = st.session_state.get("theme_mode", "dark")
                tokens = get_theme_tokens(theme_mode)

                chart_config = {
                    "background": "transparent",
                    "view": {"stroke": "transparent"},
                    "axis": {
                        "labelColor": tokens["chart_text"],
                        "titleColor": tokens["chart_text"],
                        "gridColor": tokens["chart_grid"],
                        "domainColor": tokens["chart_grid"]
                    },
                    "legend": {
                        "labelColor": tokens["chart_text"],
                        "titleColor": tokens["chart_text"]
                    }
                }

                chart_time = alt.Chart(df_time).transform_fold(
                    ['Helmets', 'Violations'],
                    as_=['Category', 'Count']
                ).mark_line(point=True).encode(
                    x=alt.X('Time (s):Q', title='Video Elapsed Time (Seconds)'),
                    y=alt.Y('Count:Q', title='Detections per Frame'),
                    color=alt.Color('Category:N', scale=alt.Scale(
                        domain=['Helmets', 'Violations'],
                        range=[tokens["success_green"], tokens["violation_red"]]
                    ))
                ).properties(height=260).configure(**chart_config)
                st.altair_chart(chart_time, use_container_width=True)

            # Violation Snapshots Gallery
            if violation_events:
                st.markdown("##### Violation Snapshots Captured")
                v_cols = st.columns(min(4, len(violation_events)))
                for idx, v_item in enumerate(violation_events[-8:]):
                    col = v_cols[idx % len(v_cols)]
                    with col:
                        if os.path.exists(v_item["path"]):
                            st.image(v_item["path"], caption=f"Time: {v_item['timestamp']} (ID: {v_item['track_id']})", use_container_width=True)

            # Exports
            col_v_exp1, col_v_exp2 = st.columns(2)
            with col_v_exp1:
                csv_v = pd.DataFrame(timeline_records).to_csv(index=False).encode("utf-8")
                st.download_button("Download Video Timeline CSV", csv_v, f"video_timeline_{int(time.time())}.csv", "text/csv", use_container_width=True)
            with col_v_exp2:
                json_v = json.dumps(timeline_records, indent=2).encode("utf-8")
                st.download_button("Download Video Timeline JSON", json_v, f"video_timeline_{int(time.time())}.json", "application/json", use_container_width=True)
