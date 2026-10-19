"""
Design system theme tokens and stylesheet loader for Helmet Detection & Road Safety Analyzer.
Provides a Vibrant Premium mobility/traffic-tech color system supporting Light and Dark modes.
"""

import os
import streamlit as st

# ==============================================================================
# DESIGN SYSTEM COLOR PALETTES
# ==============================================================================

LIGHT_TOKENS = {
    # Foundations & Surfaces
    "canvas": "#F2F6FA",
    "surface": "#FFFFFF",
    "surface_subtle": "#E8F0F5",
    "surface_hover": "#EDF4FA",
    "surface_elevated": "#FFFFFF",
    # Typography
    "text_primary": "#15314B",
    "text_secondary": "#607286",
    "text_muted": "#8898AA",
    # Borders & Dividers
    "border": "#D7E1EA",
    "border_light": "#E5ECF2",
    "border_hover": "#B8CAD9",
    # Brand & Interactive
    "primary_teal": "#087E8B",
    "primary_hover": "#066872",
    "primary_accent": "#087E8B",
    "primary_subtle": "#DDF5F2",
    "cyan_accent": "#23B8B2",
    "soft_teal": "#DDF5F2",
    "amber_accent": "#F2A541",
    "violation_red": "#D94841",
    "success_green": "#218C70",
    # Semantic Status Containers
    "compliant_bg": "#E6F7F2",
    "compliant_border": "#B3E8DC",
    "violation_bg": "#FDF0EF",
    "violation_border": "#F8C8C5",
    "warning_bg": "#FEF6EB",
    "warning_border": "#FBDCAE",
    "driver_neutral": "#475569",
    "driver_bg": "#F1F5F9",
    "driver_border": "#CBD5E1",
    # Visualizations & Charts
    "chart_bg": "transparent",
    "chart_grid": "#E2E8F0",
    "chart_text": "#607286",
    # Shadows & Elevation
    "shadow_sm": "0 1px 3px rgba(21, 49, 75, 0.05)",
    "shadow_md": "0 3px 8px rgba(21, 49, 75, 0.07), 0 1px 3px rgba(21, 49, 75, 0.04)",
    "shadow_lg": "0 8px 20px rgba(21, 49, 75, 0.09), 0 2px 6px rgba(21, 49, 75, 0.04)",
}

DARK_TOKENS = {
    # Foundations & Surfaces
    "canvas": "#0B1220",
    "surface": "#111D30",
    "surface_subtle": "#142237",
    "surface_hover": "#192B42",
    "surface_elevated": "#1B2D43",
    # Typography
    "text_primary": "#EDF4FA",
    "text_secondary": "#A5B5C6",
    "text_muted": "#758A9E",
    # Borders & Dividers
    "border": "#2B3C52",
    "border_light": "#1E2C3D",
    "border_hover": "#3D526D",
    # Brand & Interactive
    "primary_teal": "#35C6C0",
    "primary_hover": "#67DBD5",
    "primary_accent": "#35C6C0",
    "primary_subtle": "#163B40",
    "cyan_accent": "#67DBD5",
    "soft_teal": "#163B40",
    "amber_accent": "#FFBD59",
    "violation_red": "#FF7068",
    "success_green": "#52D8A4",
    # Semantic Status Containers
    "compliant_bg": "#0F2F29",
    "compliant_border": "#1A5A4D",
    "violation_bg": "#331618",
    "violation_border": "#5C2023",
    "warning_bg": "#332412",
    "warning_border": "#5C3E1B",
    "driver_neutral": "#94A3B8",
    "driver_bg": "#172334",
    "driver_border": "#2B3C52",
    # Visualizations & Charts
    "chart_bg": "transparent",
    "chart_grid": "#1E2C3D",
    "chart_text": "#A5B5C6",
    # Shadows & Elevation
    "shadow_sm": "0 1px 3px rgba(0, 0, 0, 0.35)",
    "shadow_md": "0 3px 8px rgba(0, 0, 0, 0.45), 0 1px 3px rgba(0, 0, 0, 0.3)",
    "shadow_lg": "0 8px 24px rgba(0, 0, 0, 0.55), 0 2px 8px rgba(0, 0, 0, 0.35)",
}

THEMES = {
    "light": LIGHT_TOKENS,
    "dark": DARK_TOKENS
}

def get_theme_tokens(theme_mode: str = "dark") -> dict:
    """Returns the design token dictionary for the requested theme mode."""
    mode = theme_mode.lower() if theme_mode else "dark"
    return THEMES.get(mode, DARK_TOKENS)

def get_class_colors(theme_mode: str = "dark") -> dict:
    """Returns detection bounding box colors mapped to the active theme."""
    tokens = get_theme_tokens(theme_mode)
    return {
        "helmet": tokens["success_green"],
        "no-helmet": tokens["violation_red"],
        "bicyclist": tokens["amber_accent"],
        "driver": tokens["driver_neutral"]
    }

# Backward-compatible global constants
CANVAS = LIGHT_TOKENS["canvas"]
SURFACE = LIGHT_TOKENS["surface"]
SURFACE_SUBTLE = LIGHT_TOKENS["surface_subtle"]
TEXT_PRIMARY = LIGHT_TOKENS["text_primary"]
TEXT_SECONDARY = LIGHT_TOKENS["text_secondary"]
BORDER = LIGHT_TOKENS["border"]
PRIMARY_ACCENT = LIGHT_TOKENS["primary_accent"]
PRIMARY_HOVER = LIGHT_TOKENS["primary_hover"]
PRIMARY_SUBTLE = LIGHT_TOKENS["primary_subtle"]
COMPLIANT_HELMET = LIGHT_TOKENS["success_green"]
VIOLATION_NO_HELMET = LIGHT_TOKENS["violation_red"]
WARNING_CYCLIST = LIGHT_TOKENS["amber_accent"]
DRIVER_NEUTRAL = LIGHT_TOKENS["driver_neutral"]

CLASS_COLORS = {
    "helmet": COMPLIANT_HELMET,
    "no-helmet": VIOLATION_NO_HELMET,
    "bicyclist": WARNING_CYCLIST,
    "driver": DRIVER_NEUTRAL
}

# ==============================================================================
# DYNAMIC STYLESHEET LOADER
# ==============================================================================
def load_theme(theme_mode: str = "dark"):
    """
    Reads the external CSS stylesheet from ui/styles.css and injects it
    alongside the active theme's CSS custom properties.
    """
    tokens = get_theme_tokens(theme_mode)
    css_vars = f"""
    :root {{
        --canvas:           {tokens["canvas"]};
        --surface:          {tokens["surface"]};
        --surface-subtle:   {tokens["surface_subtle"]};
        --surface-hover:    {tokens["surface_hover"]};
        --surface-elevated: {tokens["surface_elevated"]};
        --text-primary:     {tokens["text_primary"]};
        --text-secondary:   {tokens["text_secondary"]};
        --text-muted:       {tokens["text_muted"]};
        --border:           {tokens["border"]};
        --border-light:     {tokens["border_light"]};
        --border-hover:     {tokens["border_hover"]};
        --primary-teal:     {tokens["primary_teal"]};
        --primary-hover:    {tokens["primary_hover"]};
        --primary-accent:   {tokens["primary_teal"]};
        --primary-subtle:   {tokens["primary_subtle"]};
        --cyan-accent:      {tokens["cyan_accent"]};
        --soft-teal:        {tokens["soft_teal"]};
        --amber-accent:     {tokens["amber_accent"]};
        --violation-red:    {tokens["violation_red"]};
        --success-green:    {tokens["success_green"]};
        --compliant-green:  {tokens["success_green"]};
        --compliant-bg:     {tokens["compliant_bg"]};
        --compliant-border: {tokens["compliant_border"]};
        --violation-bg:     {tokens["violation_bg"]};
        --violation-border: {tokens["violation_border"]};
        --warning-amber:    {tokens["amber_accent"]};
        --warning-bg:       {tokens["warning_bg"]};
        --warning-border:   {tokens["warning_border"]};
        --driver-neutral:   {tokens["driver_neutral"]};
        --driver-bg:        {tokens["driver_bg"]};
        --driver-border:    {tokens["driver_border"]};
        --chart-bg:         {tokens["chart_bg"]};
        --chart-grid:       {tokens["chart_grid"]};
        --chart-text:       {tokens["chart_text"]};
        --shadow-sm:        {tokens["shadow_sm"]};
        --shadow-md:        {tokens["shadow_md"]};
        --shadow-lg:        {tokens["shadow_lg"]};
        --radius-sm:        6px;
        --radius-md:        10px;
        --radius-lg:        14px;
    }}
    """
    css_vars = css_vars.replace(";", " !important;")

    css_dir = os.path.dirname(__file__)
    css_path = os.path.join(css_dir, "styles.css")
    patch_path = os.path.join(css_dir, "contrast_patch.css")

    css_content = ""
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
    else:
        st.warning(f"Theme stylesheet not found at: {css_path}")

    patch_content = ""
    if os.path.exists(patch_path):
        with open(patch_path, "r", encoding="utf-8") as f:
            patch_content = f.read()

    st.markdown(
        f"<style>\n{css_content}\n{patch_content}\n{css_vars}\n</style>",
        unsafe_allow_html=True
    )
