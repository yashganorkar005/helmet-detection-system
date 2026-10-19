"""
Reusable presentation components for Helmet Detection & Road Safety Analyzer.
Provides pure HTML/Streamlit render helpers adhering to Apple-inspired minimal UI principles.
Zero emojis used across all components.
"""

import streamlit as st
from ui.icons import icon_shield, icon_info, icon_check, icon_alert

def render_page_header(
    title: str = "Helmet Detection & Road Safety Analyzer",
    subtitle: str = "Deep Learning Computer Vision Platform · YOLOv8 & ByteTrack Multi-Object Intelligence",
    badge_text: str = "System Online · Local Inference",
    badge_active: bool = True
):
    """Renders the top branding header card with an interactive theme toggle switch."""
    from ui.icons import icon_sun, icon_moon

    # Initialize theme state if not present
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "light"

    current_theme = st.session_state.theme_mode
    next_theme = "light" if current_theme == "dark" else "dark"
    btn_label = "Light Mode" if current_theme == "dark" else "Dark Mode"
    btn_icon_svg = icon_sun(size=16) if current_theme == "dark" else icon_moon(size=16)

    shield_svg = icon_shield(size=24)

    # Layout using columns inside a styled wrapper
    st.markdown('<div class="app-header">', unsafe_allow_html=True)
    col_brand, col_status, col_toggle = st.columns([5.5, 2.2, 1.3], vertical_alignment="center")

    with col_brand:
        brand_html = (
            f'<div class="app-brand">'
            f'<div class="app-brand-icon">{shield_svg}</div>'
            f'<div>'
            f'<h1 class="app-title">{title}</h1>'
            f'<p class="app-subtitle">{subtitle}</p>'
            f'</div>'
            f'</div>'
        )
        st.markdown(brand_html, unsafe_allow_html=True)

    with col_status:
        badge_html = (
            f'<div class="app-meta-badge">'
            f'<span class="status-dot"></span>'
            f'<span>{badge_text}</span>'
            f'</div>'
        )
        st.markdown(badge_html, unsafe_allow_html=True)

    with col_toggle:
        # Interactive button triggering session state update
        st.markdown('<div class="theme-toggle-wrap">', unsafe_allow_html=True)
        if st.button(btn_label, key="theme_toggle_btn", use_container_width=True, help=f"Switch to {next_theme.capitalize()} Mode"):
            st.session_state.theme_mode = next_theme
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def render_stat_card(
    label: str,
    value: str | int,
    subtext: str = None,
    tone: str = "primary",
    icon_svg: str = None
) -> str:
    """Returns HTML for an individual minimal stat card."""
    icon_html = f'<div class="stat-icon-wrapper">{icon_svg}</div>' if icon_svg else ''
    subtext_html = f'<div class="stat-subtext">{subtext}</div>' if subtext else ''
    return (
        f'<div class="stat-card tone-{tone}">'
        f'<div class="stat-header">'
        f'<span class="stat-label">{label}</span>'
        f'{icon_html}'
        f'</div>'
        f'<div class="stat-value">{value}</div>'
        f'{subtext_html}'
        f'</div>'
    )

def render_section_header(title: str, subtitle: str = None, icon_svg: str = None):
    """Renders a clean section title with optional subtitle."""
    icon_html = f"{icon_svg} " if icon_svg else ""
    st.markdown(
        f"""
        <div class="ui-section-header">
            <div class="ui-section-title">
                {icon_html}<span>{title}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if subtitle:
        st.markdown(
            f'<p style="color: var(--text-secondary); font-size: 0.85rem; margin-top: -0.5rem; margin-bottom: 1rem;">{subtitle}</p>',
            unsafe_allow_html=True
        )

def render_callout(text: str, title: str = None, tone: str = "default"):
    """Renders an informational callout container."""
    title_html = f"<strong>{title}</strong><br>" if title else ""
    cls = f"callout-box {tone}" if tone != "default" else "callout-box"
    st.markdown(
        f"""
        <div class="{cls}">
            {title_html}{text}
        </div>
        """,
        unsafe_allow_html=True
    )

def render_empty_state(title: str, description: str, icon_svg: str = None):
    """Renders a minimal neutral placeholder empty state."""
    icon_html = f'<div class="empty-state-icon">{icon_svg}</div>' if icon_svg else ''
    st.markdown(
        f"""
        <div class="empty-state">
            {icon_html}
            <div class="empty-state-title">{title}</div>
            <div class="empty-state-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_badge(text: str, category: str = "neutral") -> str:
    """Returns HTML for a small semantic tag/badge."""
    cls_map = {
        "compliant": "badge-compliant",
        "violation": "badge-violation",
        "warning": "badge-warning",
        "neutral": "badge-neutral"
    }
    css_cls = cls_map.get(category.lower(), "badge-neutral")
    return f'<span class="badge {css_cls}">{text}</span>'
