"""
Custom label renderer for Helmet Detection & Road Safety Analyzer.

Replaces the label-drawing pass of Ultralytics results[0].plot() with a
conflict-aware version that enforces vertical priority:
    helmet / no-helmet  >>>  driver / bicyclist

All visual style (font, scale, thickness, colors, background rectangles)
exactly matches the Ultralytics Annotator so the output is indistinguishable
from the built-in renderer except when labels would otherwise overlap.

Usage (all four inference paths):
    from ui.annotations import plot_with_priority_labels
    annotated = plot_with_priority_labels(results, img_bgr, show_boxes, show_labels, show_conf)
"""

import cv2
import numpy as np
from ultralytics.utils.plotting import colors as ult_colors

# ---------------------------------------------------------------------------
# Priority table: higher number = must be drawn higher (further from top)
# ---------------------------------------------------------------------------
_LABEL_PRIORITY = {
    "helmet":    2,
    "no-helmet": 2,
    "driver":    1,
    "bicyclist": 1,
}

# Overlap check: two labels are "near enough to conflict" when their
# anchor boxes share horizontal space AND their label slots would overlap.
_OVERLAP_IOU_X_THRESHOLD = 0.20   # fraction of the narrower box width


def _lw_sf_tf(img_bgr: np.ndarray):
    """Replicate Ultralytics line-width / font-scale / font-thickness formula."""
    h, w, c = img_bgr.shape
    lw = max(round((h + w + c) / 2 * 0.003), 2)
    tf = max(lw - 1, 1)
    sf = lw / 3
    return lw, sf, tf


def _label_slot(x1: int, y1: int, label: str, sf: float, tf: int, img_h: int):
    """
    Return the bounding rectangle of a single label drawn at (x1, y1).

    Returns (lx1, ly1, lx2, ly2) in pixel coords where:
        ly1 = top of background rect
        ly2 = bottom of background rect  (== y1, i.e. box top edge)
    """
    (tw, th), _ = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)
    th += 3                          # Ultralytics adds 3px padding
    outside = y1 >= th               # fits above box top edge
    if outside:
        rect_y1 = y1 - th
        rect_y2 = y1
    else:
        rect_y1 = y1
        rect_y2 = y1 + th
    return (x1, rect_y1, x1 + tw, rect_y2, th, outside)


def _rects_overlap(a, b):
    """True when two (x1,y1,x2,y2,...) label slot rects intersect."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[2], a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def _draw_label(img, x1, anchor_y, label, sf, tf, color, img_h):
    """
    Draw label background + text at (x1, anchor_y) exactly matching Ultralytics style.
    anchor_y is the bottom edge of the background rectangle.
    """
    (tw, th), _ = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)
    th += 3
    # Clamp x so label never runs off the right edge
    if x1 + tw > img.shape[1]:
        x1 = max(0, img.shape[1] - tw)
    outside = anchor_y >= th
    if outside:
        p2 = (x1 + tw, anchor_y - th)
        text_y = anchor_y - 2
    else:
        p2 = (x1 + tw, anchor_y + th)
        text_y = anchor_y + th - 1
    cv2.rectangle(img, (x1, anchor_y), p2, color, -1, cv2.LINE_AA)
    # Choose text colour: white on dark, black on light (matches Ultralytics get_txt_color)
    r, g, b = color
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    txt_color = (0, 0, 0) if luma > 160 else (255, 255, 255)
    cv2.putText(img, label, (x1, text_y), 0, sf, txt_color, thickness=tf, lineType=cv2.LINE_AA)


def plot_with_priority_labels(
    results,
    show_boxes: bool = True,
    show_labels: bool = True,
    show_conf: bool = True,
) -> np.ndarray:
    """
    Render detection results with conflict-aware label positioning.

    Parameters
    ----------
    results : ultralytics Results object (from model.predict / model.track)
    show_boxes : draw bounding-box rectangles
    show_labels : draw class name labels
    show_conf : append confidence score to label text

    Returns
    -------
    np.ndarray  BGR image (same dtype / shape as the input frame)
    """
    # Step 1 ─ draw boxes (and colours) using the built-in renderer,
    # but suppress labels so we control them entirely.
    annotated = results.plot(
        boxes=show_boxes,
        labels=False,        # we redraw these ourselves
        conf=False,
    )

    if not show_labels:
        return annotated

    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return annotated

    img_h, img_w = annotated.shape[:2]
    lw, sf, tf = _lw_sf_tf(annotated)

    # -----------------------------------------------------------------------
    # Step 2 ─ build per-detection label info, sorted high-priority first
    # -----------------------------------------------------------------------
    # Pre-extract tracking IDs once so we can look them up by box index.
    # boxes.is_track is True only when model.track() was used; boxes.id is
    # a (N,1) tensor of integer IDs in that case, otherwise None.
    track_ids = None
    if getattr(boxes, "is_track", False) and boxes.id is not None:
        track_ids = boxes.id.int().tolist()   # list of ints, one per detection

    detections = []
    for i, box in enumerate(boxes):
        cls_id  = int(box.cls[0])
        cname   = results.names[cls_id].lower()
        conf    = float(box.conf[0])
        xyxy    = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        color   = ult_colors(cls_id, True)          # BGR

        # Build label: "id:N class conf" when tracking, "class conf" otherwise.
        # track_ids is a flat list so index i directly maps to this detection.
        tid = track_ids[i] if (track_ids is not None and i < len(track_ids)) else None
        id_prefix = f"id:{tid} " if tid is not None else ""
        if show_conf:
            label = f"{id_prefix}{cname} {conf:.2f}"
        else:
            label = f"{id_prefix}{cname}"

        prio    = _LABEL_PRIORITY.get(cname, 1)
        detections.append({
            "cls": cname, "label": label, "color": color, "prio": prio,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })

    # Sort so that higher-priority labels are positioned first
    detections.sort(key=lambda d: d["prio"], reverse=True)

    # -----------------------------------------------------------------------
    # Step 3 ─ assign each label an anchor_y (bottom edge of label rect),
    #           resolving conflicts greedily from highest to lowest priority.
    # -----------------------------------------------------------------------
    placed = []   # list of (x1, anchor_y, lx1, ly1, lx2, ly2, det)

    for det in detections:
        x1, y1 = det["x1"], det["y1"]
        label   = det["label"]

        # Initial anchor: top of box (Ultralytics default)
        anchor_y = y1

        slot    = _label_slot(x1, anchor_y, label, sf, tf, img_h)
        th      = slot[4]   # padded text height

        # Check against already-placed labels; push up if they overlap
        for _ in range(30):   # max 30 nudge iterations → avoids infinite loop
            slot = _label_slot(x1, anchor_y, label, sf, tf, img_h)
            conflict = False
            for (px1, pa_y, plx1, ply1, plx2, ply2, pdet) in placed:
                pslot = (plx1, ply1, plx2, ply2)
                cslot = (slot[0], slot[1], slot[2], slot[3])
                if _rects_overlap(cslot, pslot):
                    # Determine push direction based on priority
                    if det["prio"] >= pdet["prio"]:
                        # We're equal or higher: the other one was placed first,
                        # so we need to move ourselves away from the box top.
                        # Move this label upward (further from the box).
                        anchor_y = ply1   # sit our bottom edge at their top
                    else:
                        # We're lower priority, move us down past the other label
                        anchor_y = ply2 + th
                    conflict = True
                    break
            if not conflict:
                break

        # Clamp so label stays within image bounds (top)
        slot = _label_slot(x1, anchor_y, label, sf, tf, img_h)
        if slot[1] < 0:
            anchor_y = th   # push down to fit

        slot = _label_slot(x1, anchor_y, label, sf, tf, img_h)
        placed.append((x1, anchor_y, slot[0], slot[1], slot[2], slot[3], det))

    # -----------------------------------------------------------------------
    # Step 4 ─ draw all labels at their resolved positions
    # -----------------------------------------------------------------------
    for (x1, anchor_y, *_, det) in placed:
        _draw_label(annotated, x1, anchor_y, det["label"], sf, tf, det["color"], img_h)

    return annotated
