"""
Reusable professional SVG icons for Helmet Detection & Road Safety Analyzer.
All icons use consistent stroke width, sizing, accessible SVG attributes, and avoid emoji.
"""

def _svg_wrap(svg_content: str, size: int = 20, color: str = None, extra_class: str = "") -> str:
    stroke_attr = f'stroke="{color}"' if color else 'stroke="currentColor"'
    style_attr = f'style="color: {color};"' if color else ''
    cls_attr = f'class="svg-icon {extra_class}"' if extra_class else 'class="svg-icon"'
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'{stroke_attr} {style_attr} {cls_attr} stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" '
        f'aria-hidden="true">{svg_content}</svg>'
    )

def icon_shield(size: int = 24, color: str = None) -> str:
    """Road safety shield mark."""
    return _svg_wrap(
        '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'
        '<path d="M9 12l2 2 4-4"/>',
        size, color
    )

def icon_image(size: int = 20, color: str = None) -> str:
    """Image analysis icon."""
    return _svg_wrap(
        '<rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>'
        '<circle cx="8.5" cy="8.5" r="1.5"/>'
        '<polyline points="21 15 16 10 5 21"/>',
        size, color
    )

def icon_layers(size: int = 20, color: str = None) -> str:
    """Batch processing layers icon."""
    return _svg_wrap(
        '<polygon points="12 2 2 7 12 12 22 7 12 2"/>'
        '<polyline points="2 17 12 22 22 17"/>'
        '<polyline points="2 12 12 17 22 12"/>',
        size, color
    )

def icon_video(size: int = 20, color: str = None) -> str:
    """Video analysis icon."""
    return _svg_wrap(
        '<polygon points="23 7 16 12 23 17 23 7"/>'
        '<rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>',
        size, color
    )

def icon_camera(size: int = 20, color: str = None) -> str:
    """Live camera stream icon."""
    return _svg_wrap(
        '<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>'
        '<circle cx="12" cy="13" r="4"/>',
        size, color
    )

def icon_chart(size: int = 20, color: str = None) -> str:
    """Analytics chart icon."""
    return _svg_wrap(
        '<line x1="18" y1="20" x2="18" y2="10"/>'
        '<line x1="12" y1="20" x2="12" y2="4"/>'
        '<line x1="6" y1="20" x2="6" y2="14"/>',
        size, color
    )

def icon_settings(size: int = 20, color: str = None) -> str:
    """Settings / control panel icon."""
    return _svg_wrap(
        '<circle cx="12" cy="12" r="3"/>'
        '<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>',
        size, color
    )

def icon_download(size: int = 18, color: str = None) -> str:
    """Download action icon."""
    return _svg_wrap(
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        '<polyline points="7 10 12 15 17 10"/>'
        '<line x1="12" y1="15" x2="12" y2="3"/>',
        size, color
    )

def icon_refresh(size: int = 16, color: str = None) -> str:
    """Refresh / reset icon."""
    return _svg_wrap(
        '<polyline points="23 4 23 10 17 10"/>'
        '<polyline points="1 20 1 14 7 14"/>'
        '<path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>',
        size, color
    )

def icon_trash(size: int = 16, color: str = None) -> str:
    """Delete / trash icon."""
    return _svg_wrap(
        '<polyline points="3 6 5 6 21 6"/>'
        '<path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>'
        '<line x1="10" y1="11" x2="10" y2="17"/>'
        '<line x1="14" y1="11" x2="14" y2="17"/>',
        size, color
    )

def icon_check(size: int = 18, color: str = None) -> str:
    """Check / compliance icon."""
    return _svg_wrap(
        '<polyline points="20 6 9 17 4 12"/>',
        size, color
    )

def icon_alert(size: int = 18, color: str = None) -> str:
    """Alert / violation triangle icon."""
    return _svg_wrap(
        '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
        '<line x1="12" y1="9" x2="12" y2="13"/>'
        '<line x1="12" y1="17" x2="12.01" y2="17"/>',
        size, color
    )

def icon_info(size: int = 18, color: str = None) -> str:
    """Information circle icon."""
    return _svg_wrap(
        '<circle cx="12" cy="12" r="10"/>'
        '<line x1="12" y1="16" x2="12" y2="12"/>'
        '<line x1="12" y1="8" x2="12.01" y2="8"/>',
        size, color
    )

def icon_upload(size: int = 24, color: str = None) -> str:
    """Upload cloud/arrow icon."""
    return _svg_wrap(
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        '<polyline points="17 8 12 3 7 8"/>'
        '<line x1="12" y1="3" x2="12" y2="15"/>',
        size, color
    )

def icon_play(size: int = 18, color: str = None) -> str:
    """Play icon."""
    return _svg_wrap(
        '<polygon points="5 3 19 12 5 21 5 3"/>',
        size, color
    )

def icon_user(size: int = 18, color: str = None) -> str:
    """User / rider icon."""
    return _svg_wrap(
        '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>'
        '<circle cx="12" cy="7" r="4"/>',
        size, color
    )

def icon_target(size: int = 18, color: str = None) -> str:
    """Target / ROI icon."""
    return _svg_wrap(
        '<circle cx="12" cy="12" r="10"/>'
        '<circle cx="12" cy="12" r="6"/>'
        '<circle cx="12" cy="12" r="2"/>',
        size, color
    )

def icon_sun(size: int = 18, color: str = None) -> str:
    """Sun / light mode toggle icon."""
    return _svg_wrap(
        '<circle cx="12" cy="12" r="5"/>'
        '<line x1="12" y1="1" x2="12" y2="3"/>'
        '<line x1="12" y1="21" x2="12" y2="23"/>'
        '<line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>'
        '<line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>'
        '<line x1="1" y1="12" x2="3" y2="12"/>'
        '<line x1="21" y1="12" x2="23" y2="12"/>'
        '<line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>'
        '<line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>',
        size, color
    )

def icon_moon(size: int = 18, color: str = None) -> str:
    """Moon / dark mode toggle icon."""
    return _svg_wrap(
        '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>',
        size, color
    )
