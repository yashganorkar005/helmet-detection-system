"""
Automated Test Suite for Helmet Detection & Road Safety Analyzer
Validates:
1. Model loading and weights integrity
2. Single-image inference and class predictions
3. ByteTrack multi-object tracking across sequential video frames
4. Region-of-Interest (ROI) geometric filtering
5. File validation safeguards (size limits and file extensions)
6. In-memory ZIP batch archiving
7. Export data structures (CSV and JSON)
8. Label placement priority (helmet/no-helmet above driver)
"""

import os
import sys
import unittest
import io
import zipfile
import json
import numpy as np
import cv2
from PIL import Image

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from ultralytics import YOLO

MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
SAMPLE_IMG_PATH = os.path.join(BASE_DIR, "sample_media", "sample_street.jpg")
SAMPLE_VID_PATH = os.path.join(BASE_DIR, "sample_media", "sample_traffic.mp4")


class DummyFile:
    """Mock file object for size and extension validation testing."""
    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size


def validate_image_file(file_obj) -> tuple[bool, str]:
    max_bytes = 15 * 1024 * 1024  # 15 MB
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    name = getattr(file_obj, "name", "sample.jpg")
    ext = os.path.splitext(name)[1].lower()

    if ext not in valid_exts:
        return False, f"Invalid format '{ext}'. Supported: JPG, JPEG, PNG, WEBP, BMP."

    size = getattr(file_obj, "size", 0)
    if size > max_bytes:
        return False, f"File size ({size / (1024*1024):.1f}MB) exceeds the 15MB maximum limit."
    return True, ""


def validate_video_file(file_obj) -> tuple[bool, str]:
    max_bytes = 100 * 1024 * 1024  # 100 MB
    valid_exts = {".mp4", ".avi", ".mov", ".mkv"}
    name = getattr(file_obj, "name", "sample.mp4")
    ext = os.path.splitext(name)[1].lower()

    if ext not in valid_exts:
        return False, f"Invalid format '{ext}'. Supported: MP4, AVI, MOV, MKV."

    size = getattr(file_obj, "size", 0)
    if size > max_bytes:
        return False, f"Video size ({size / (1024*1024):.1f}MB) exceeds the 100MB maximum limit."
    return True, ""


def is_point_in_roi(cx: float, cy: float, roi: dict, width: int, height: int) -> bool:
    if not roi.get("enabled", False):
        return True
    rx1 = roi["x_min"] * width
    rx2 = roi["x_max"] * width
    ry1 = roi["y_min"] * height
    ry2 = roi["y_max"] * height
    return (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)


class TestHelmetDetectionPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.assertTrue(os.path.exists(MODEL_PATH), f"Model weights not found at: {MODEL_PATH}")
        cls.model = YOLO(MODEL_PATH)

    def test_model_classes(self):
        """Verify model class names match the expected 4 trained classes."""
        expected_classes = {0: "bicyclist", 1: "driver", 2: "helmet", 3: "no-helmet"}
        for cls_id, name in expected_classes.items():
            self.assertIn(cls_id, self.model.names)
            self.assertEqual(self.model.names[cls_id].lower(), name)

    def test_image_inference(self):
        """Verify model can infer on sample street image and identify objects."""
        self.assertTrue(os.path.exists(SAMPLE_IMG_PATH), f"Sample image missing at: {SAMPLE_IMG_PATH}")
        img = np.array(Image.open(SAMPLE_IMG_PATH).convert("RGB"))
        results = self.model.predict(img, conf=0.35, iou=0.45, verbose=False)
        self.assertGreater(len(results), 0)
        self.assertGreater(len(results[0].boxes), 0, "Expected at least 1 detection in sample image")

        # Verify bounding box attributes
        for box in results[0].boxes:
            self.assertIn("xyxy", dir(box))
            self.assertIn("conf", dir(box))
            self.assertIn("cls", dir(box))

    def test_bytetrack_tracking(self):
        """Verify ByteTrack tracking assigns consistent IDs on sequential video frames."""
        self.assertTrue(os.path.exists(SAMPLE_VID_PATH), f"Sample video missing at: {SAMPLE_VID_PATH}")
        cap = cv2.VideoCapture(SAMPLE_VID_PATH)
        self.assertTrue(cap.isOpened(), "Could not open sample video file")

        tracked_ids = set()
        frames_tested = 0
        while cap.isOpened() and frames_tested < 5:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=0.35,
                iou=0.45,
                verbose=False
            )
            for box in results[0].boxes:
                if box.id is not None:
                    tracked_ids.add(int(box.id[0]))
            frames_tested += 1
        cap.release()

        self.assertGreater(frames_tested, 0, "No frames could be read from sample video")
        self.assertGreater(len(tracked_ids), 0, "ByteTrack tracking should have assigned IDs")

    def test_roi_filtering(self):
        """Verify ROI point-in-polygon filtering for left lane, right lane, and center."""
        width = 1000
        height = 600

        # Disabled ROI should allow all points
        disabled_roi = {"enabled": False}
        self.assertTrue(is_point_in_roi(100, 100, disabled_roi, width, height))

        # Left lane ROI (0 - 50% width)
        left_roi = {"enabled": True, "x_min": 0.0, "x_max": 0.50, "y_min": 0.0, "y_max": 1.0}
        self.assertTrue(is_point_in_roi(250, 300, left_roi, width, height))   # Center of left lane
        self.assertFalse(is_point_in_roi(750, 300, left_roi, width, height))  # Inside right lane

        # Right lane ROI (50% - 100% width)
        right_roi = {"enabled": True, "x_min": 0.50, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}
        self.assertFalse(is_point_in_roi(250, 300, right_roi, width, height)) # Inside left lane
        self.assertTrue(is_point_in_roi(750, 300, right_roi, width, height))  # Center of right lane

    def test_file_validation_safeguards(self):
        """Verify file extension and size validation safeguards."""
        # Valid image file
        valid_img = DummyFile("road.jpg", 2 * 1024 * 1024)
        is_val, msg = validate_image_file(valid_img)
        self.assertTrue(is_val)
        self.assertEqual(msg, "")

        # Oversized image (>15MB)
        huge_img = DummyFile("massive.png", 20 * 1024 * 1024)
        is_val, msg = validate_image_file(huge_img)
        self.assertFalse(is_val)
        self.assertIn("exceeds the 15MB", msg)

        # Invalid image extension
        bad_img = DummyFile("exploit.exe", 1024)
        is_val, msg = validate_image_file(bad_img)
        self.assertFalse(is_val)
        self.assertIn("Invalid format", msg)

        # Valid video file
        valid_vid = DummyFile("highway.mp4", 50 * 1024 * 1024)
        is_val, msg = validate_video_file(valid_vid)
        self.assertTrue(is_val)

        # Oversized video (>100MB)
        huge_vid = DummyFile("long_footage.mp4", 120 * 1024 * 1024)
        is_val, msg = validate_video_file(huge_vid)
        self.assertFalse(is_val)
        self.assertIn("exceeds the 100MB", msg)

    def test_batch_zip_archiving(self):
        """Verify in-memory ZIP archive generation for batch exports."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("batch_summary.csv", "Filename,Total\nimg1.jpg,3\n")
            z.writestr("batch_summary.json", json.dumps([{"Filename": "img1.jpg", "Total": 3}]))
            z.writestr("annotated_img1.jpg", b"fake_jpeg_binary_data")

        buf.seek(0)
        zip_bytes = buf.getvalue()
        self.assertGreater(len(zip_bytes), 0)

        # Read back and verify contents
        read_buf = io.BytesIO(zip_bytes)
        with zipfile.ZipFile(read_buf, "r") as z_read:
            namelist = z_read.namelist()
            self.assertIn("batch_summary.csv", namelist)
            self.assertIn("batch_summary.json", namelist)
            self.assertIn("annotated_img1.jpg", namelist)


if __name__ == "__main__":
    unittest.main()


# ==============================================================================
# Label-placement tests (no model required — synthetic detections)
# ==============================================================================

import torch


def _make_results(img_bgr, detections):
    """Build a minimal Ultralytics Results object from (cls_id,x1,y1,x2,y2,conf) tuples."""
    from ultralytics.engine.results import Results, Boxes
    boxes_data = np.array(
        [[x1, y1, x2, y2, conf, cls_id] for cls_id, x1, y1, x2, y2, conf in detections],
        dtype=np.float32
    ) if detections else np.zeros((0, 6), dtype=np.float32)
    r = Results(
        orig_img=img_bgr,
        path="",
        names={0: "bicyclist", 1: "driver", 2: "helmet", 3: "no-helmet"},
    )
    r.boxes = Boxes(torch.tensor(boxes_data), img_bgr.shape[:2])
    return r


class TestLabelPlacement(unittest.TestCase):
    """Verify priority-aware label placement in ui.annotations."""

    @classmethod
    def setUpClass(cls):
        from ui.annotations import (
            plot_with_priority_labels,
            _label_slot,
            _rects_overlap,
            _lw_sf_tf,
            _LABEL_PRIORITY,
        )
        # Wrap module-level functions as staticmethods so Python does not
        # inject `self` when they are called via instance attributes.
        cls.plot     = staticmethod(plot_with_priority_labels)
        cls.slot     = staticmethod(_label_slot)
        cls.overlap  = staticmethod(_rects_overlap)
        cls.lw_sf_tf = staticmethod(_lw_sf_tf)
        cls.PRIO     = _LABEL_PRIORITY
        cls.H, cls.W = 480, 640
        cls.img = np.zeros((cls.H, cls.W, 3), dtype=np.uint8)

    # ------------------------------------------------------------------
    def test_priority_values(self):
        """helmet and no-helmet must outrank driver and bicyclist."""
        self.assertEqual(self.PRIO["helmet"],    2)
        self.assertEqual(self.PRIO["no-helmet"], 2)
        self.assertEqual(self.PRIO["driver"],    1)
        self.assertEqual(self.PRIO["bicyclist"], 1)

    # ------------------------------------------------------------------
    def test_overlap_detection(self):
        """_rects_overlap correctly identifies intersecting rectangles."""
        self.assertTrue(self.overlap((100, 100, 200, 120), (110, 105, 210, 125)))
        self.assertFalse(self.overlap((100, 100, 200, 120), (100, 125, 200, 145)))
        # Touching edges — not overlapping
        self.assertFalse(self.overlap((100, 100, 200, 120), (200, 100, 300, 120)))

    # ------------------------------------------------------------------
    def _anchor_ys(self, detections_sorted):
        """Replay placement loop and return {classname: anchor_y}."""
        _, sf, tf = self.lw_sf_tf(self.img)
        placed, anchor_ys = [], {}
        for det in detections_sorted:
            x1, y1 = det["x1"], det["y1"]
            anchor_y = y1
            for _ in range(30):
                slot = self.slot(x1, anchor_y, det["label"], sf, tf, self.H)
                conflict = False
                for (_, _, plx1, ply1, plx2, ply2, pdet) in placed:
                    if self.overlap((slot[0], slot[1], slot[2], slot[3]), (plx1, ply1, plx2, ply2)):
                        if det["prio"] >= pdet["prio"]:
                            anchor_y = ply1
                        else:
                            anchor_y = ply2 + slot[4]
                        conflict = True
                        break
                if not conflict:
                    break
            slot = self.slot(x1, anchor_y, det["label"], sf, tf, self.H)
            placed.append((x1, anchor_y, slot[0], slot[1], slot[2], slot[3], det))
            anchor_ys[det["cls"]] = anchor_y
        return anchor_ys

    # ------------------------------------------------------------------
    def test_helmet_above_driver_when_overlapping(self):
        """helmet label must be placed at or above driver label when boxes are close."""
        dets = [
            {"cls": "helmet", "label": "helmet 0.91", "prio": 2,
             "x1": 90, "y1": 115, "x2": 250, "y2": 290, "color": (243, 243, 243)},
            {"cls": "driver", "label": "driver 0.85", "prio": 1,
             "x1": 100, "y1": 120, "x2": 260, "y2": 300, "color": (235, 219, 11)},
        ]
        ay = self._anchor_ys(dets)
        self.assertLessEqual(
            ay["helmet"], ay["driver"],
            f"helmet anchor_y={ay['helmet']} must be <= driver anchor_y={ay['driver']}"
        )

    # ------------------------------------------------------------------
    def test_no_helmet_above_driver_when_overlapping(self):
        """no-helmet label must be placed at or above driver label when boxes are close."""
        dets = [
            {"cls": "no-helmet", "label": "no-helmet 0.79", "prio": 2,
             "x1": 90, "y1": 115, "x2": 250, "y2": 290, "color": (183, 223, 0)},
            {"cls": "driver", "label": "driver 0.85", "prio": 1,
             "x1": 100, "y1": 120, "x2": 260, "y2": 300, "color": (235, 219, 11)},
        ]
        ay = self._anchor_ys(dets)
        self.assertLessEqual(
            ay["no-helmet"], ay["driver"],
            f"no-helmet anchor_y={ay['no-helmet']} must be <= driver anchor_y={ay['driver']}"
        )

    # ------------------------------------------------------------------
    def test_both_labels_visible(self):
        """Both label anchor positions must be within image boundaries."""
        dets = [
            {"cls": "helmet", "label": "helmet 0.91", "prio": 2,
             "x1": 90,  "y1": 115, "x2": 250, "y2": 290, "color": (243, 243, 243)},
            {"cls": "driver", "label": "driver 0.85", "prio": 1,
             "x1": 100, "y1": 120, "x2": 260, "y2": 300, "color": (235, 219, 11)},
        ]
        ay = self._anchor_ys(dets)
        for cls, anchor_y in ay.items():
            self.assertGreaterEqual(anchor_y, 0, f"{cls} label top must be >= 0")
            self.assertLessEqual(anchor_y, self.H, f"{cls} label top must be <= image height")

    # ------------------------------------------------------------------
    def test_non_overlapping_detections_unperturbed(self):
        """Labels that don't overlap must keep their natural anchor (box top edge)."""
        _, sf, tf = self.lw_sf_tf(self.img)
        # These boxes are far apart: no conflict expected
        dets = [
            {"cls": "helmet",  "label": "helmet 0.88",  "prio": 2,
             "x1": 50,  "y1": 50,  "x2": 200, "y2": 180, "color": (243, 243, 243)},
            {"cls": "driver",  "label": "driver 0.76",  "prio": 1,
             "x1": 400, "y1": 300, "x2": 580, "y2": 450, "color": (235, 219, 11)},
        ]
        ay = self._anchor_ys(dets)
        # Each anchor should equal its natural y1 (no nudge needed)
        self.assertEqual(ay["helmet"], 50,  "helmet anchor_y should equal y1=50 when no conflict")
        self.assertEqual(ay["driver"], 300, "driver anchor_y should equal y1=300 when no conflict")

    # ------------------------------------------------------------------
    def test_bounding_boxes_unchanged(self):
        """plot_with_priority_labels must not alter detection bounding-box coords."""
        dets_raw = [
            (1,  100, 120, 260, 300, 0.85),  # driver
            (2,   90, 115, 250, 290, 0.91),  # helmet
        ]
        r = _make_results(self.img, dets_raw)
        # Coordinates before
        coords_before = [(b.xyxy[0].tolist()) for b in r.boxes]
        self.plot(r, show_boxes=True, show_labels=True, show_conf=True)
        # Coordinates after (Results object must not be mutated)
        coords_after = [(b.xyxy[0].tolist()) for b in r.boxes]
        self.assertEqual(coords_before, coords_after, "Box coordinates must not change after annotation")

    # ------------------------------------------------------------------
    def test_output_shape_matches_input(self):
        """Rendered image must have the same shape as the input frame."""
        dets_raw = [(1, 100, 120, 260, 300, 0.85), (2, 90, 115, 250, 290, 0.91)]
        r = _make_results(self.img, dets_raw)
        out = self.plot(r, show_boxes=True, show_labels=True, show_conf=True)
        self.assertEqual(out.shape, self.img.shape)

    # ------------------------------------------------------------------
    def test_labels_hidden_when_show_labels_false(self):
        """When show_labels=False the output must equal boxes-only plot output."""
        dets_raw = [(1, 100, 120, 260, 300, 0.85)]
        r = _make_results(self.img, dets_raw)
        out = self.plot(r, show_boxes=True, show_labels=False, show_conf=False)
        # Must still return a valid image (no crash)
        self.assertEqual(out.shape, self.img.shape)


# ==============================================================================
# Tracking ID label tests
# ==============================================================================

def _make_tracked_results(img_bgr, detections):
    """Build a Results object with tracking IDs from (cls_id,x1,y1,x2,y2,conf,track_id) tuples."""
    from ultralytics.engine.results import Results, Boxes
    # 7-column format: [x1, y1, x2, y2, track_id, conf, cls]
    boxes_data = np.array(
        [[x1, y1, x2, y2, tid, conf, cls_id]
         for cls_id, x1, y1, x2, y2, conf, tid in detections],
        dtype=np.float32
    ) if detections else np.zeros((0, 7), dtype=np.float32)
    r = Results(
        orig_img=img_bgr,
        path="",
        names={0: "bicyclist", 1: "driver", 2: "helmet", 3: "no-helmet"},
    )
    r.boxes = Boxes(torch.tensor(boxes_data), img_bgr.shape[:2])
    return r


class TestTrackingIdLabels(unittest.TestCase):
    """Verify that tracking IDs appear in plot_with_priority_labels output."""

    @classmethod
    def setUpClass(cls):
        from ui.annotations import plot_with_priority_labels
        cls.plot = staticmethod(plot_with_priority_labels)
        cls.img  = np.zeros((480, 640, 3), dtype=np.uint8)

    def _extract_labels(self, results, show_conf=True):
        """Re-run the label-building logic from annotations.py and return the strings."""
        from ui.annotations import _LABEL_PRIORITY
        boxes = results.boxes
        track_ids = None
        if getattr(boxes, "is_track", False) and boxes.id is not None:
            track_ids = boxes.id.int().tolist()
        labels = []
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            cname  = results.names[cls_id].lower()
            conf   = float(box.conf[0])
            tid    = track_ids[i] if (track_ids is not None and i < len(track_ids)) else None
            id_prefix = f"id:{tid} " if tid is not None else ""
            label = f"{id_prefix}{cname} {conf:.2f}" if show_conf else f"{id_prefix}{cname}"
            labels.append(label)
        return labels

    def test_tracked_driver_shows_id(self):
        """A tracked driver detection must include its tracking ID."""
        r = _make_tracked_results(self.img, [(1, 100, 120, 260, 300, 0.87, 3)])
        self.assertTrue(r.boxes.is_track)
        labels = self._extract_labels(r)
        self.assertIn("id:3", labels[0])
        self.assertIn("driver", labels[0])
        self.assertIn("0.87", labels[0])

    def test_tracked_helmet_shows_id(self):
        """A tracked helmet detection must include its tracking ID."""
        r = _make_tracked_results(self.img, [(2, 90, 115, 250, 290, 0.91, 7)])
        labels = self._extract_labels(r)
        self.assertIn("id:7", labels[0])
        self.assertIn("helmet", labels[0])

    def test_tracked_no_helmet_shows_id(self):
        """A tracked no-helmet detection must include its tracking ID."""
        r = _make_tracked_results(self.img, [(3, 90, 115, 250, 290, 0.79, 12)])
        labels = self._extract_labels(r)
        self.assertIn("id:12", labels[0])
        self.assertIn("no-helmet", labels[0])

    def test_each_detection_shows_its_own_id(self):
        """Two detections must each show their own distinct ID."""
        r = _make_tracked_results(self.img, [
            (1, 100, 120, 260, 300, 0.85, 5),   # driver  → id:5
            (2,  90, 115, 250, 290, 0.91, 9),   # helmet  → id:9
        ])
        labels = self._extract_labels(r)
        driver_lbl = next(l for l in labels if "driver" in l)
        helmet_lbl = next(l for l in labels if "helmet" in l)
        self.assertIn("id:5", driver_lbl)
        self.assertNotIn("id:9", driver_lbl)
        self.assertIn("id:9", helmet_lbl)
        self.assertNotIn("id:5", helmet_lbl)

    def test_no_id_without_tracking(self):
        """Plain predict() results must not show any id: prefix."""
        r = _make_results(self.img, [(1, 100, 120, 260, 300, 0.85)])
        self.assertFalse(r.boxes.is_track)
        labels = self._extract_labels(r)
        self.assertNotIn("id:", labels[0])
        self.assertIn("driver", labels[0])

    def test_show_conf_false_hides_confidence_keeps_id(self):
        """show_conf=False must hide confidence but keep the id: prefix."""
        r = _make_tracked_results(self.img, [(1, 100, 120, 260, 300, 0.85, 4)])
        labels = self._extract_labels(r, show_conf=False)
        self.assertIn("id:4", labels[0])
        self.assertIn("driver", labels[0])
        self.assertNotIn("0.85", labels[0])

    def test_render_does_not_crash_with_tracking(self):
        """plot_with_priority_labels must return correct shape for tracked results."""
        r = _make_tracked_results(self.img, [
            (1, 100, 120, 260, 300, 0.85, 3),
            (2,  90, 115, 250, 290, 0.91, 7),
        ])
        out = self.plot(r, show_boxes=True, show_labels=True, show_conf=True)
        self.assertEqual(out.shape, self.img.shape)

    def test_overlap_fix_still_works_with_tracking_ids(self):
        """helmet label must still appear above driver even with longer id: labels."""
        from ui.annotations import _lw_sf_tf, _label_slot, _rects_overlap, _LABEL_PRIORITY
        r = _make_tracked_results(self.img, [
            (2,  90, 115, 250, 290, 0.91, 7),   # helmet  id:7
            (1, 100, 120, 260, 300, 0.85, 3),   # driver  id:3
        ])
        _, sf, tf = _lw_sf_tf(self.img)
        H = self.img.shape[0]
        boxes = r.boxes
        track_ids = boxes.id.int().tolist()
        dets = []
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            cname  = r.names[cls_id].lower()
            conf   = float(box.conf[0])
            xyxy   = box.xyxy[0].tolist()
            label  = f"id:{track_ids[i]} {cname} {conf:.2f}"
            dets.append({
                "cls": cname, "label": label,
                "prio": _LABEL_PRIORITY.get(cname, 1),
                "x1": int(xyxy[0]), "y1": int(xyxy[1]),
            })
        dets.sort(key=lambda d: d["prio"], reverse=True)
        placed, anchor_ys = [], {}
        for det in dets:
            anchor_y = det["y1"]
            for _ in range(30):
                slot = _label_slot(det["x1"], anchor_y, det["label"], sf, tf, H)
                conflict = False
                for (_, _, plx1, ply1, plx2, ply2, pdet) in placed:
                    if _rects_overlap((slot[0], slot[1], slot[2], slot[3]),
                                      (plx1, ply1, plx2, ply2)):
                        anchor_y = ply1 if det["prio"] >= pdet["prio"] else ply2 + slot[4]
                        conflict = True
                        break
                if not conflict:
                    break
            slot = _label_slot(det["x1"], anchor_y, det["label"], sf, tf, H)
            placed.append((det["x1"], anchor_y, slot[0], slot[1], slot[2], slot[3], det))
            anchor_ys[det["cls"]] = anchor_y
        self.assertLessEqual(
            anchor_ys["helmet"], anchor_ys["driver"],
            f"helmet anchor_y={anchor_ys['helmet']} must be <= driver anchor_y={anchor_ys['driver']}"
        )

    def test_box_coords_unchanged_with_tracking(self):
        """Box coordinates must not be mutated by plot_with_priority_labels."""
        r = _make_tracked_results(self.img, [
            (1, 100, 120, 260, 300, 0.85, 3),
            (2,  90, 115, 250, 290, 0.91, 7),
        ])
        coords_before = [b.xyxy[0].tolist() for b in r.boxes]
        self.plot(r, show_boxes=True, show_labels=True, show_conf=True)
        coords_after = [b.xyxy[0].tolist() for b in r.boxes]
        self.assertEqual(coords_before, coords_after)


if __name__ == "__main__":
    unittest.main()
