"""
Road Safety Analytics page module for Helmet Detection & Road Safety Analyzer.
Displays aggregated compliance metrics, Altair charts for class distribution and confidence histograms,
plus the saved violation snapshot archive.
"""

import os
import pandas as pd
import altair as alt
import streamlit as st
from ui.theme import get_theme_tokens
from ui.components import render_callout

def render_analytics_page(violation_dir: str):
    """Renders the comprehensive road safety analytics dashboard."""
    st.markdown(
        """
        <p style="color: var(--text-secondary); font-size: 0.86rem; margin-bottom: 1.25rem;">
            Comprehensive statistical road-safety intelligence aggregated across all evaluated images, batches, and video feeds.
        </p>
        """,
        unsafe_allow_html=True
    )

    # Statistical Counting Rules Explainer
    render_callout(
        """
        • <strong>Raw Bounding Boxes:</strong> Instantaneous object detections detected in individual video frames or images.<br>
        • <strong>Unique Tracked Entities:</strong> Individual physical riders tracked continuously across sequential frames using ByteTrack IDs.<br>
        • <strong>Violation Events:</strong> Recorded instances of non-compliance where a rider was confirmed without a certified helmet.<br>
        • <strong>Compliance Rate:</strong> <code>(Compliant Helmets / Total Evaluated Riders) × 100</code>.
        """,
        title="Statistical Counting Rules & Scientific Methodology",
        tone="info"
    )

    # High-level Metrics Row
    c_m1, c_m2, c_m3, c_m4, c_m5 = st.columns(5)
    with c_m1:
        st.metric("Frames Analyzed", st.session_state.frames_analyzed)
    with c_m2:
        st.metric("Raw Box Detections", st.session_state.total_detections)
    with c_m3:
        st.metric("Unique Tracked Riders", len(st.session_state.unique_tracked_ids))
    with c_m4:
        st.metric("Violations Flagged", st.session_state.violation_count)
    with c_m5:
        tot = st.session_state.total_detections
        h_cnt = st.session_state.helmet_count
        c_pct = f"{(h_cnt / tot * 100):.1f}%" if tot > 0 else "N/A"
        st.metric("Cumulative Compliance", c_pct)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Visualizations Row: Class Distribution + Confidence Scores
    col_chart1, col_chart2 = st.columns(2)

    # Retrieve active theme tokens for dynamic chart styling
    theme_mode = st.session_state.get("theme_mode", "dark")
    tokens = get_theme_tokens(theme_mode)

    # Altair theme configuration
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

    with col_chart1:
        st.markdown("##### Class Distribution of Detections")
        class_df = pd.DataFrame([
            {"Class": "Helmet", "Count": st.session_state.class_counts["helmet"], "Category": "Compliant"},
            {"Class": "No-Helmet", "Count": st.session_state.class_counts["no-helmet"], "Category": "Violation"},
            {"Class": "Driver", "Count": st.session_state.class_counts["driver"], "Category": "Operator"},
            {"Class": "Bicyclist", "Count": st.session_state.class_counts["bicyclist"], "Category": "Cyclist"}
        ])

        chart_classes = alt.Chart(class_df).mark_bar(cornerRadius=5).encode(
            x=alt.X('Class:N', title='Road User Class', sort='-y'),
            y=alt.Y('Count:Q', title='Detections Count'),
            color=alt.Color('Class:N', scale=alt.Scale(
                domain=['Helmet', 'No-Helmet', 'Driver', 'Bicyclist'],
                range=[
                    tokens["success_green"],
                    tokens["violation_red"],
                    tokens["driver_neutral"],
                    tokens["amber_accent"]
                ]
            ), legend=None),
            tooltip=['Class', 'Count', 'Category']
        ).properties(height=280).configure(**chart_config)
        st.altair_chart(chart_classes, use_container_width=True)

    with col_chart2:
        st.markdown("##### Model Confidence Score Distribution")
        if st.session_state.confidence_scores:
            conf_df = pd.DataFrame({"Confidence": st.session_state.confidence_scores})
            chart_conf = alt.Chart(conf_df).mark_bar(cornerRadius=4, color=tokens["primary_teal"]).encode(
                x=alt.X('Confidence:Q', bin=alt.Bin(maxbins=15), title='Prediction Confidence (0.0 - 1.0)'),
                y=alt.Y('count()', title='Frequency of Detections')
            ).properties(height=280).configure(**chart_config)
            st.altair_chart(chart_conf, use_container_width=True)
        else:
            st.info("Run image or video analysis to populate confidence distribution telemetry.")

    # Saved Violations Gallery
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Violation Image Archive")
    target_dir = getattr(st.session_state, "session_violation_dir", violation_dir)
    if os.path.exists(target_dir):
        saved_violations = [f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    else:
        saved_violations = []

    if saved_violations:
        st.markdown(f"Found **{len(saved_violations)}** violation snapshot(s) in local storage:")
        v_display_cols = st.columns(min(4, len(saved_violations)))
        for idx, v_name in enumerate(saved_violations[-8:]):
            v_col = v_display_cols[idx % len(v_display_cols)]
            with v_col:
                v_path = os.path.join(target_dir, v_name)
                st.image(v_path, caption=v_name, use_container_width=True)
    else:
        st.info("No violation snapshots saved yet.")
