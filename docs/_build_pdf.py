"""
_build_pdf.py — Generate docs/deployment-architecture.pdf

Usage (from project root):
    python docs/_build_pdf.py

Requires: reportlab (pip install reportlab)
All other imports are stdlib.

Output: docs/deployment-architecture.pdf  (A4 landscape, 3 pages)
"""

import os
import sys

# ---------------------------------------------------------------------------
# Ensure reportlab is available before proceeding
# ---------------------------------------------------------------------------
try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import mm, cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether, PageBreak,
    )
    from reportlab.graphics.shapes import (
        Drawing, Rect, Line, String, Group, Polygon,
    )
    from reportlab.graphics import renderPDF
    from reportlab.graphics.shapes import ArcPath
except ImportError:
    print("ERROR: reportlab is not installed.  Run:  pip install reportlab")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
NAVY       = colors.HexColor("#0D2137")
TEAL       = colors.HexColor("#087E8B")
BLUE       = colors.HexColor("#1769AA")
LIGHT_BLUE = colors.HexColor("#DDE8F5")
LIGHT_GREY = colors.HexColor("#F2F6FA")
MID_GREY   = colors.HexColor("#CBD5E1")
STEEL      = colors.HexColor("#457B9D")
TEXT       = colors.HexColor("#15314B")
MUTED      = colors.HexColor("#6B7C99")
DARK_PANEL = colors.HexColor("#132B45")
PANEL_TEXT = colors.HexColor("#8DB4CC")
RED        = colors.HexColor("#D94841")
GREEN      = colors.HexColor("#218C70")
WHITE      = colors.white

# ---------------------------------------------------------------------------
# Paths & page geometry
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "deployment-architecture.pdf")

PAGE_W, PAGE_H = landscape(A4)          # 841.89 x 595.28 pt
LEFT_M = RIGHT_M = 20 * mm              # 56.69 pt each
TOP_M  = 18 * mm                        # 51.02 pt
BOT_M  = 16 * mm                        # 45.35 pt
CONTENT_W = PAGE_W - LEFT_M - RIGHT_M  # 728.50 pt
FRAME_H   = PAGE_H - TOP_M - BOT_M     # 498.91 pt

# ---------------------------------------------------------------------------
# Styles  — tightened to let all content fit in 3 pages
# ---------------------------------------------------------------------------
_base = getSampleStyleSheet()

def _style(name, parent="Normal", **kw):
    return ParagraphStyle(name, parent=_base[parent], **kw)

H1      = _style("H1",     fontSize=18, textColor=NAVY,  fontName="Helvetica-Bold",
                 spaceAfter=3, leading=22)
SUBTTL  = _style("Sub",    fontSize=9,  textColor=MUTED, fontName="Helvetica",
                 spaceAfter=0, leading=12)
H2      = _style("H2",     fontSize=11, textColor=TEAL,  fontName="Helvetica-Bold",
                 spaceAfter=2, leading=14, spaceBefore=6)
H3      = _style("H3",     fontSize=9,  textColor=BLUE,  fontName="Helvetica-Bold",
                 spaceAfter=2, leading=12, spaceBefore=4)
BODY    = _style("Body",   fontSize=8,  textColor=TEXT,  fontName="Helvetica",
                 leading=11, spaceAfter=2)
CAPTION = _style("Caption",fontSize=7,  textColor=MUTED, fontName="Helvetica",
                 leading=9,  alignment=TA_CENTER)
BULLET  = _style("Bullet", fontSize=8,  textColor=TEXT,  fontName="Helvetica",
                 leading=11, leftIndent=10, firstLineIndent=-7, spaceAfter=1)
TH      = _style("TH",     fontSize=8,  textColor=WHITE, fontName="Helvetica-Bold",
                 leading=10, alignment=TA_CENTER)
TD      = _style("TD",     fontSize=8,  textColor=TEXT,  fontName="Helvetica",
                 leading=10)

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _arrow(d, x1, y1, x2, y2, col=NAVY, width=1.2, dash=None):
    import math
    ln = Line(x1, y1, x2, y2, strokeColor=col, strokeWidth=width)
    if dash:
        ln.strokeDashArray = dash
    d.add(ln)
    angle = math.atan2(y2 - y1, x2 - x1)
    sz = 4.5
    ax = x2 - sz * math.cos(angle - 0.4)
    ay = y2 - sz * math.sin(angle - 0.4)
    bx = x2 - sz * math.cos(angle + 0.4)
    by = y2 - sz * math.sin(angle + 0.4)
    d.add(Polygon([x2, y2, ax, ay, bx, by],
                  fillColor=col, strokeColor=col, strokeWidth=0))


def _box(d, x, y, w, h, fill=WHITE, stroke=NAVY, sw=1.2, rx=3):
    d.add(Rect(x, y, w, h, rx=rx, ry=rx,
               fillColor=fill, strokeColor=stroke, strokeWidth=sw))


def _label(d, x, y, text, size=8, col=TEXT, bold=False, anchor="middle"):
    d.add(String(x, y, text,
                 fontSize=size, fillColor=col,
                 fontName="Helvetica-Bold" if bold else "Helvetica",
                 textAnchor=anchor))


# ---------------------------------------------------------------------------
# Architecture diagram — sized to fit the remaining space on page 1.
#
# Measured available height after title + exec overview + arch heading +
# arch intro paragraph + spacers + caption = ~325 pt.
# We use DH = 295 to leave ~30 pt breathing room.
# Width = CONTENT_W (728.5 pt), matching the text column exactly.
# All original x-coordinates (designed for DW=780) are scaled by S=DW/780.
# ---------------------------------------------------------------------------
def build_arch_diagram():
    DW = CONTENT_W          # 728.5 pt — fills content column exactly
    DH = 278                # diagram + caption (9pt) = 287pt; available = 299pt
    S  = DW / 780.0         # ≈ 0.9340 — uniform horizontal scale

    d = Drawing(DW, DH)
    d.add(Rect(0, 0, DW, DH, fillColor=LIGHT_GREY, strokeColor=None))

    lane_h = DH / 3          # ≈ 98.3 pt each

    # Lane backgrounds
    d.add(Rect(0, 0,          DW, lane_h, fillColor=colors.HexColor("#EEF2F7"),
               strokeColor=MID_GREY, strokeWidth=0.5))
    d.add(Rect(0, lane_h,     DW, lane_h, fillColor=NAVY,
               strokeColor=MID_GREY, strokeWidth=0.5))
    d.add(Rect(0, lane_h * 2, DW, lane_h, fillColor=LIGHT_BLUE,
               strokeColor=MID_GREY, strokeWidth=0.5))

    # Coloured sidebar strips
    sw = round(14 * S)
    d.add(Rect(0, lane_h * 2, sw, lane_h, fillColor=BLUE,   strokeColor=None))
    d.add(Rect(0, lane_h,     sw, lane_h, fillColor=TEAL,   strokeColor=None))
    d.add(Rect(0, 0,          sw, lane_h, fillColor=STEEL,  strokeColor=None))

    # Lane labels
    lx = sw + 3
    _label(d, lx, lane_h * 3 - 6,  "Developer / Source",   size=7, col=BLUE,  bold=True, anchor="start")
    _label(d, lx, lane_h * 2 - 6,  "Deployment / Runtime", size=7, col=TEAL,  bold=True, anchor="start")
    _label(d, lx, lane_h - 6,       "User / Browser",       size=7, col=STEEL, bold=True, anchor="start")

    # Scale helper
    def sx(v): return round(v * S, 1)

    # Component height (proportional to DH)
    COMP_H = round(50 * DH / 280)

    # ── LANE 1: Developer / Source  (top) ──────────────────────────────────
    L1_Y     = lane_h * 2
    COMP_TOP = L1_Y + (lane_h - COMP_H) / 2

    # Developer Workstation
    bx, bw = sx(22), sx(90)
    _box(d, bx, COMP_TOP, bw, COMP_H, fill=WHITE, stroke=BLUE)
    _label(d, bx+bw/2, COMP_TOP+COMP_H-10, "Developer",       size=8, col=NAVY, bold=True)
    _label(d, bx+bw/2, COMP_TOP+COMP_H-21, "Workstation",     size=8, col=NAVY)
    _label(d, bx+bw/2, COMP_TOP+6,          "Local Python 3.12", size=7, col=MUTED)

    # Arrow Dev → GitHub
    ax1, ax2 = bx+bw, sx(148)
    _arrow(d, ax1, COMP_TOP+COMP_H/2, ax2, COMP_TOP+COMP_H/2, col=BLUE)
    _label(d, (ax1+ax2)/2, COMP_TOP+COMP_H/2+5, "git push", size=7, col=BLUE)

    # GitHub Repository
    gbx, gbw = sx(148), sx(145)
    _box(d, gbx, COMP_TOP-8, gbw, COMP_H+16, fill=WHITE, stroke=BLUE)
    _label(d, gbx+gbw/2, COMP_TOP+COMP_H+2, "GitHub Repository", size=8, col=NAVY, bold=True)
    for i, txt in enumerate([
        "app.py  |  models/best.pt (5.94 MB)",
        "requirements.txt  |  packages.txt",
        ".streamlit/config.toml",
        "ui/  |  tests/  |  sample_media/",
    ]):
        _label(d, gbx+4, COMP_TOP+COMP_H-10-i*10, txt, size=6.5, col=MUTED, anchor="start")

    # Arrow GitHub → Build
    gx2 = sx(340)
    _arrow(d, gbx+gbw, COMP_TOP+COMP_H/2, gx2, COMP_TOP+COMP_H/2, col=TEAL, dash=[4,3])
    _label(d, (gbx+gbw+gx2)/2, COMP_TOP+COMP_H/2+5, "Deploy trigger", size=7, col=TEAL)

    # ── LANE 2: Deployment / Runtime  (middle) ──────────────────────────────
    L2_Y  = lane_h
    B_TOP = L2_Y + (lane_h - COMP_H) / 2

    # Build Step
    bldx, bldw = sx(340), sx(90)
    _box(d, bldx, B_TOP, bldw, COMP_H, fill=DARK_PANEL, stroke=TEAL)
    bm = bldx + bldw/2
    _label(d, bm, B_TOP+COMP_H-10, "Community Cloud",  size=7.5, col=WHITE, bold=True)
    _label(d, bm, B_TOP+COMP_H-21, "Build",            size=7.5, col=WHITE, bold=True)
    _label(d, bm, B_TOP+15,         "pip install + apt-get", size=6.5, col=PANEL_TEXT)
    _label(d, bm, B_TOP+6,          "packages.txt",          size=6.5, col=PANEL_TEXT)

    # Arrow Build → Running App
    bax1, bax2 = bldx+bldw, sx(460)
    _arrow(d, bax1, B_TOP+COMP_H/2, bax2, B_TOP+COMP_H/2, col=TEAL)
    _label(d, (bax1+bax2)/2, B_TOP+COMP_H/2+5, "Start", size=7, col=PANEL_TEXT)

    # Running App box — right edge = DW - 2
    app_x  = sx(460)
    app_w  = DW - app_x - 2
    app_y  = L2_Y + 4
    app_h  = lane_h - 8
    _box(d, app_x, app_y, app_w, app_h, fill=DARK_PANEL, stroke=TEAL, sw=1.5)
    _label(d, app_x+app_w/2, app_y+app_h-7,
           "Running Streamlit App  (app.py)", size=8, col=WHITE, bold=True)

    sub_y = app_y + 4
    sub_h = app_h - 18
    sub_w = (app_w - 12) / 3

    sub1_x = app_x + 4
    _box(d, sub1_x, sub_y, sub_w, sub_h, fill=NAVY, stroke=BLUE, sw=0.8)
    _label(d, sub1_x+sub_w/2, sub_y+sub_h-7, "UI Layer",
           size=7, col=colors.HexColor("#60ABDC"), bold=True)
    for i, t in enumerate(["ui/theme.py", "ui/layout.py",
                            "ui/components.py", "5 page modules", "styles.css"]):
        _label(d, sub1_x+3, sub_y+sub_h-18-i*9, t, size=6, col=PANEL_TEXT, anchor="start")

    sub2_x = sub1_x + sub_w + 4
    _box(d, sub2_x, sub_y, sub_w, sub_h, fill=NAVY, stroke=TEAL, sw=0.8)
    _label(d, sub2_x+sub_w/2, sub_y+sub_h-7, "Inference Layer",
           size=7, col=colors.HexColor("#60ABDC"), bold=True)
    for i, t in enumerate(["YOLOv8 (4 classes)", "model.predict/track",
                            "ByteTrack", "annotations.py", "priority labels"]):
        _label(d, sub2_x+3, sub_y+sub_h-18-i*9, t, size=6, col=PANEL_TEXT, anchor="start")

    sub3_x = sub2_x + sub_w + 4
    _box(d, sub3_x, sub_y, sub_w, sub_h, fill=NAVY, stroke=RED, sw=0.8)
    _label(d, sub3_x+sub_w/2, sub_y+sub_h-7, "Session State",
           size=7, col=colors.HexColor("#60ABDC"), bold=True)
    for i, t in enumerate(["uuid4 per session", "violations/<id>/",
                            "queue.Queue()", "_video_temp_path",
                            "@st.cache_resource"]):
        _label(d, sub3_x+3, sub_y+sub_h-18-i*9, t, size=6, col=PANEL_TEXT, anchor="start")

    # ── LANE 3: User / Browser  (bottom) ────────────────────────────────────
    U_TOP = (lane_h - COMP_H) / 2

    # Browser box
    brx, brw = sx(22), sx(78)
    _box(d, brx, U_TOP, brw, COMP_H, fill=WHITE, stroke=STEEL)
    _label(d, brx+brw/2, U_TOP+COMP_H-10, "Browser",          size=8, col=NAVY, bold=True)
    _label(d, brx+brw/2, U_TOP+11,         "HTTPS / WebSocket", size=7, col=MUTED)

    # Arrow Browser → App (up)
    ba1, ba2 = brx+brw, app_x
    mx = (ba1+ba2)/2
    _arrow(d, ba1, U_TOP+COMP_H*0.6, ba2, L2_Y+lane_h/2, col=BLUE)
    _label(d, mx-20, U_TOP+COMP_H+5, "HTTPS request", size=7, col=BLUE)

    # Arrow App → Browser (down, results)
    _arrow(d, app_x+2, L2_Y+lane_h*0.38, ba1+2, U_TOP+COMP_H*0.78, col=GREEN)
    _label(d, mx-30, L2_Y-5, "annotated results + downloads", size=7, col=GREEN)

    # 5 Tabs box
    tabs_x, tabs_w = sx(190), sx(275)
    _box(d, tabs_x, U_TOP-4, tabs_w, COMP_H+8, fill=WHITE, stroke=STEEL)
    _label(d, tabs_x+tabs_w/2, U_TOP+COMP_H+2, "5 Application Tabs",
           size=8, col=NAVY, bold=True)
    tab_labels = ["Single Image", "Batch", "Video", "Live Camera", "Analytics"]
    tab_cols   = [BLUE, BLUE, BLUE, GREEN, BLUE]
    tw = round(tabs_w / 5) - 4
    for i, (tl, tc) in enumerate(zip(tab_labels, tab_cols)):
        tx = tabs_x + 4 + i * (tw + 3)
        d.add(Rect(tx, U_TOP+6, tw, 17, rx=3, ry=3,
                   fillColor=LIGHT_BLUE if tc==BLUE else colors.HexColor("#D4EDDA"),
                   strokeColor=tc, strokeWidth=0.8))
        _label(d, tx+tw/2, U_TOP+16, tl, size=6, col=NAVY)

    # WebRTC box — right-justified to DW
    webrtc_x = tabs_x + tabs_w + 7
    webrtc_w = DW - webrtc_x - 2
    _box(d, webrtc_x, U_TOP-4, webrtc_w, COMP_H+8, fill=WHITE, stroke=RED, sw=1, rx=4)
    d.add(Rect(webrtc_x, U_TOP-4, webrtc_w, COMP_H+8, rx=4, ry=4,
               fillColor=WHITE, strokeColor=RED, strokeWidth=1,
               strokeDashArray=[4, 3]))
    _label(d, webrtc_x+webrtc_w/2, U_TOP+COMP_H+2, "WebRTC (Live Camera)",
           size=7, col=RED, bold=True)
    for i, t in enumerate(["getUserMedia", "WebRTC peer conn.",
                            "STUN: google:19302", "streamlit-webrtc",
                            "av.VideoFrame"]):
        _label(d, webrtc_x+4, U_TOP+COMP_H-8-i*9, t, size=6.5, col=MUTED, anchor="start")

    # Arrow: Live Camera tab → WebRTC box
    live_i    = 3
    live_tx   = tabs_x + 4 + live_i * (tw + 3) + tw
    _arrow(d, live_tx, U_TOP+14, webrtc_x, U_TOP+14, col=RED, dash=[3,3])

    return d


# ---------------------------------------------------------------------------
# Page template (header + footer drawn on canvas, not in flowable story)
# ---------------------------------------------------------------------------
def _header_footer(canvas, doc):
    canvas.saveState()
    W, H = landscape(A4)
    canvas.setStrokeColor(TEAL)
    canvas.setLineWidth(1.5)
    canvas.line(20*mm, H-14*mm, W-20*mm, H-14*mm)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(NAVY)
    canvas.drawString(20*mm, H-11*mm,
                      "Helmet Detection & Road Safety Analyzer — Deployment Architecture")
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(W-20*mm, H-11*mm, "Confidential — Internal Use")
    canvas.setStrokeColor(MID_GREY)
    canvas.setLineWidth(0.75)
    canvas.line(20*mm, 10*mm, W-20*mm, 10*mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawCentredString(W/2, 6*mm, f"Page {doc.page}")
    canvas.drawString(20*mm, 6*mm, "Python 3.12 · YOLOv8 8.4.182 · Streamlit 1.66.0")
    canvas.drawRightString(W-20*mm, 6*mm, "Streamlit Community Cloud (intended)")
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Table factory
# ---------------------------------------------------------------------------
def _make_table(header_row, data_rows, col_widths, alt_fill=None):
    if alt_fill is None:
        alt_fill = colors.HexColor("#EEF2F7")
    rows = [[Paragraph(str(c), TH) for c in header_row]]
    for row in data_rows:
        rows.append([Paragraph(str(c), TD) for c in row])
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0), NAVY),
        ("TEXTCOLOR",    (0,0), (-1,0), WHITE),
        ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 8),
        ("ALIGN",        (0,0), (-1,-1), "LEFT"),
        ("VALIGN",       (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",   (0,0), (-1,-1), 2),
        ("BOTTOMPADDING",(0,0), (-1,-1), 2),
        ("LEFTPADDING",  (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, alt_fill]),
        ("GRID",         (0,0), (-1,-1), 0.4, MID_GREY),
        ("LINEBELOW",    (0,0), (-1,0),  1.5, TEAL),
    ]))
    return t


# ---------------------------------------------------------------------------
# Page 1: Title + Executive Overview + Architecture diagram
# ---------------------------------------------------------------------------
def build_page1():
    story = []

    story.append(Paragraph("Helmet Detection &amp; Road Safety Analyzer", H1))
    story.append(Paragraph(
        "Deployment Architecture Reference  ·  Python 3.12 · YOLOv8 8.4.182 "
        "· Streamlit 1.66.0 · ByteTrack · Streamlit Community Cloud (intended)",
        SUBTTL))
    story.append(HRFlowable(width="100%", thickness=1.5, color=TEAL, spaceAfter=5))

    story.append(Paragraph("1. Executive Overview", H2))
    story.append(Paragraph(
        "Road safety enforcement faces a persistent gap: manual monitoring of "
        "helmet compliance is labour-intensive, inconsistent, and non-scalable. "
        "This project provides a computer-vision pipeline that identifies "
        "motorcyclists and bicyclists, classifies their helmet status in real "
        "time, and surfaces violation evidence through a browser-based interface.",
        BODY))
    for c in [
        "<b>Single-image analysis</b> — annotated JPEG + CSV + JSON exports.",
        "<b>Batch processing</b> — multi-image upload; results packaged as a ZIP.",
        "<b>Video analysis</b> — per-frame inference, ByteTrack identity "
        "persistence, ROI masking, violation snapshots, timeline CSV/JSON.",
        "<b>Live camera (WebRTC)</b> — real-time inference from browser webcam; "
        "violation snapshots throttled to 1 per 4 seconds.",
        "<b>Session analytics</b> — Altair charts for class distribution and "
        "confidence histogram; violation archive gallery.",
    ]:
        story.append(Paragraph(f"• {c}", BULLET))

    story.append(Spacer(1, 4))
    story.append(Paragraph("2. System Architecture", H2))
    story.append(Paragraph(
        "Three horizontal layers: developer/source (GitHub), deployment/runtime "
        "(Streamlit Community Cloud), and user/browser.", BODY))
    story.append(Spacer(1, 3))
    story.append(build_arch_diagram())
    story.append(Paragraph(
        "Figure 1 — Three-layer deployment architecture. "
        "Dashed arrows = optional/conditional paths (deploy trigger, WebRTC).",
        CAPTION))

    story.append(PageBreak())
    return story


# ---------------------------------------------------------------------------
# Page 2: Input Processing Pipeline + Technology Decisions
# ---------------------------------------------------------------------------
def build_page2():
    story = []

    story.append(Paragraph("3. Input Processing Pipeline", H2))

    story.append(Paragraph("3.1  Single Image", H3))
    for step in [
        "User uploads a file (JPG / JPEG / PNG / WEBP / BMP, max 15 MB).",
        "PIL.Image opens the file; converted to a NumPy array.",
        "model.predict(conf=0.35, iou=0.45) runs synchronously.",
        "plot_with_priority_labels() renders the annotated frame "
        "(helmet/no-helmet priority=2, driver/bicyclist=1; greedy slot nudging, up to 30 iterations).",
        "Annotated array encoded as JPEG and offered for download.",
        "Detection data serialised to CSV and JSON.",
    ]:
        story.append(Paragraph(f"• {step}", BULLET))

    story.append(Paragraph("3.2  Batch Processing", H3))
    for step in [
        "User uploads multiple files (same format and size limits as single image).",
        "Per-image predict-and-annotate loop runs sequentially.",
        "Annotated JPEGs + batch_summary.csv + batch_summary.json "
        "written into an in-memory io.BytesIO ZIP archive and offered as a download.",
    ]:
        story.append(Paragraph(f"• {step}", BULLET))

    story.append(Paragraph("3.3  Video Analysis", H3))
    for step in [
        "Upload (MP4 / AVI / MOV / MKV, max 100 MB) → NamedTemporaryFile in "
        "session violation directory; path stored in session_state._video_temp_path.",
        "probe_cap reads frame count, FPS, width, height for telemetry banner.",
        "User configures frame stride (1–5), confidence, IoU, ROI preset, and "
        "tracking toggle, then clicks Start.",
        "Analysis cap iterates frames at stride. Each frame → model.track "
        "(persist=True, bytetrack.yaml) or model.predict → "
        "plot_with_priority_labels → optional ROI overlay.",
        "Violation frames: snapshot saved as JPG (max 1/s). Timeline data "
        "available as CSV/JSON after analysis. Temp file deleted in finally block.",
    ]:
        story.append(Paragraph(f"• {step}", BULLET))

    story.append(Paragraph("3.4  Live Camera (WebRTC)", H3))
    for step in [
        "webrtc_streamer() opens ICE with stun:stun.l.google.com:19302.",
        "Browser requests camera via getUserMedia; WebRTC peer connection established.",
        "VideoProcessor.recv() receives each av.VideoFrame, converts to NumPy, "
        "runs model.predict(), annotates via plot_with_priority_labels(), "
        "returns as av.VideoFrame.",
        "Violation snapshots saved to session directory, throttled to max 1 per 4 s.",
    ]:
        story.append(Paragraph(f"• {step}", BULLET))

    story.append(Spacer(1, 5))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GREY, spaceAfter=4))

    story.append(Paragraph("4. Technology Decisions", H2))

    tech_headers = ["Technology", "Role", "Benefit", "Trade-off"]
    tech_data = [
        ["YOLOv8 (ultralytics 8.4.182)", "Object detection",
         "Single-stage; native ByteTrack; simple predict/track API",
         "CPU-only inference; no annotated video download"],
        ["Streamlit 1.66.0", "Web UI framework",
         "Python-native; Community Cloud free tier; no separate frontend",
         "Limited UI customisation; threading constraints"],
        ["streamlit-webrtc 0.78.1", "Browser camera",
         "Handles WebRTC pipeline in Streamlit; no separate media server",
         "Requires webcam; TURN not configured"],
        ["ByteTrack + lap 0.5.13", "Multi-object tracking",
         "Stable id:N labels across video frames",
         "Video mode only; adds lap dependency"],
        ["PyAV 17.1.0", "Video frame codec",
         "Correct colour-space for av.VideoFrame ↔ NumPy",
         "Binary codec dependency"],
        ["OpenCV headless 5.0.0.93", "Frame I/O + ROI",
         "No GUI deps; smaller container footprint",
         "No display functions"],
        ["Pillow 12.3.0", "Image decode/encode",
         "Broad format support for single-image and batch modes", "—"],
        ["torch 2.14.1+cpu", "YOLOv8 backend",
         "No GPU required; installs as ultralytics dependency",
         "CPU inference; ~800 MB RAM on free tier"],
        ["Altair 6.3.0", "Analytics charts",
         "Declarative; renders natively in Streamlit; no extra JS bundle", "—"],
    ]

    CW = CONTENT_W
    col_ws = [CW*0.19, CW*0.12, CW*0.39, CW*0.30]
    story.append(_make_table(tech_headers, tech_data, col_ws))

    story.append(PageBreak())
    return story


# ---------------------------------------------------------------------------
# Page 3: Deployment Lifecycle + Tests + Limitations + STAR
# ---------------------------------------------------------------------------
def build_page3():
    story = []

    story.append(Paragraph("5. Deployment Lifecycle", H2))
    for i, s in enumerate([
        "Developer writes and tests code locally (Python 3.12.2).",
        "Developer pushes to GitHub via git push.",
        "Streamlit Community Cloud detects the push and triggers a build.",
        "Build installs system packages from packages.txt (libgl1, libglib2.0-0t64).",
        "Build installs Python packages from requirements.txt at pinned versions.",
        "app.py starts; @st.cache_resource loads models/best.pt once (~5.94 MB).",
        "Streamlit serves the app on a public HTTPS URL.",
        "Visitor's browser opens the URL; WebSocket-backed connection established.",
        "Visitor uses one of the five tabs; inference runs server-side; "
        "annotated results returned to the browser.",
    ], 1):
        story.append(Paragraph(f"{i}. {s}", BULLET))

    story.append(Spacer(1, 3))
    story.append(Paragraph("5.1  Session Isolation", H3))
    story.append(Paragraph(
        "Each session gets a uuid.uuid4().hex subdirectory under violations/ "
        "preventing cross-session snapshot leakage. A queue.Queue() per session "
        "handles event logging. The YOLOv8 model is shared via @st.cache_resource "
        "(read-only inference is thread-safe).", BODY))

    story.append(Paragraph("5.2  Operational Considerations", H3))
    for o in [
        "<b>Memory</b> — torch + model consume significant RAM; concurrent video "
        "sessions on the ~800 MB free tier can be memory-intensive.",
        "<b>WebRTC</b> — STUN resolves ICE for most residential networks. "
        "Corporate firewalls blocking UDP may prevent connection. No TURN configured.",
        "<b>Ephemeral storage</b> — violation snapshots and temp video files lost "
        "on process restart. Persistent storage requires an external object store.",
        "<b>File limits</b> — images 15 MB (JPG/PNG/WEBP/BMP); video 100 MB "
        "(MP4/AVI/MOV/MKV).",
    ]:
        story.append(Paragraph(f"• {o}", BULLET))

    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GREY, spaceAfter=4))

    story.append(Paragraph("6. Test Coverage", H2))
    test_headers = ["Test Class", "Tests", "What is verified"]
    test_data = [
        ["TestHelmetDetectionPipeline", "6",
         "Model loads (4-class map) · single-image inference · ByteTrack assigns "
         "IDs · ROI filtering · file-type validation · batch ZIP contents"],
        ["TestLabelPlacement", "9",
         "Priority order (helmet/no-helmet=2, driver/bicyclist=1) · overlap "
         "detection · boundary clamping · box immutability · labels-off path"],
        ["TestTrackingIdLabels", "9",
         "id:N prefix per class with tracking · distinct IDs · no prefix without "
         "tracking · show_conf=False keeps id: · render shape · coord immutability"],
        ["Total", "24", "All pass in approximately 2.5 seconds"],
    ]
    CW = CONTENT_W
    col_ws = [CW*0.27, CW*0.07, CW*0.66]
    t = _make_table(test_headers, test_data, col_ws)
    t.setStyle(TableStyle([
        ("FONTNAME",   (0,4), (-1,4), "Helvetica-Bold"),
        ("BACKGROUND", (0,4), (-1,4), colors.HexColor("#D4EDDA")),
    ]))
    story.append(t)

    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GREY, spaceAfter=4))

    story.append(Paragraph("7. Known Limitations", H2))
    for l in [
        "No annotated video download — CSV/JSON timeline only.",
        "Live camera requires a physical webcam; non-functional in headless environments.",
        "Violation archive is session-scoped (violations/<uuid>/) and lost on process restart.",
        "CPU inference only on Community Cloud free tier.",
        "Compliance rate = helmets ÷ (helmets + no-helmets); driver/bicyclist counts excluded.",
        "~800 MB free RAM on Community Cloud; video + model inference can approach this.",
        "WebRTC STUN fails behind strict corporate firewalls; TURN not configured.",
    ]:
        story.append(Paragraph(f"• {l}", BULLET))

    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GREY, spaceAfter=4))

    story.append(Paragraph("8. STAR Interview Summary", H2))
    star_data = [
        ["Situation",
         "Road safety monitoring relies on manual observers — inconsistent and "
         "non-scalable. A CV system was needed to detect helmet compliance from "
         "image, video, and live-camera inputs at low cost."],
        ["Task",
         "Build a full-stack detection app: fine-tuned YOLOv8 (4 classes), "
         "Streamlit web UI with 5 tabs, multi-session isolation, a custom priority "
         "label renderer, and a 24-test pytest suite."],
        ["Action",
         "Selected YOLOv8 for native ByteTrack integration. Built "
         "plot_with_priority_labels() with two-tier priority and greedy slot "
         "nudging (up to 30 iterations). Per-session UUID directories isolate "
         "violation snapshots. Targeted Streamlit Community Cloud to avoid "
         "server-management overhead."],
        ["Result",
         "Working 5-tab application (image, batch, video, live-camera, analytics). "
         "24 tests pass in ~2.5 s. No database, no REST API, no Docker. Primary "
         "constraint: live-camera tab requires a webcam and a WebRTC-permitting network."],
    ]
    col_ws_star = [CW*0.11, CW*0.89]
    t2 = _make_table(["", "Summary"], star_data, col_ws_star)
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0,1), (0,1), colors.HexColor("#DDE8F5")),
        ("BACKGROUND", (0,2), (0,2), colors.HexColor("#EEF2F7")),
        ("BACKGROUND", (0,3), (0,3), colors.HexColor("#DDE8F5")),
        ("BACKGROUND", (0,4), (0,4), colors.HexColor("#D4EDDA")),
        ("FONTNAME",   (0,1), (0,-1), "Helvetica-Bold"),
        ("TEXTCOLOR",  (0,1), (0,-1), NAVY),
    ]))
    story.append(t2)

    return story


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=landscape(A4),
        leftMargin=LEFT_M,
        rightMargin=RIGHT_M,
        topMargin=TOP_M,
        bottomMargin=BOT_M,
        title="Helmet Detection & Road Safety Analyzer — Deployment Architecture",
        author="Generated by _build_pdf.py",
    )
    story = build_page1() + build_page2() + build_page3()
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    print(f"PDF written to: {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
