# Helmet Detection & Road Safety Analyzer

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66.0-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-8.4.182-00B4D8.svg?style=flat)](https://github.com/ultralytics/ultralytics)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL_3.0-blue.svg)](LICENSE)

A local, privacy-first road safety analysis platform that detects helmet compliance in traffic images and video using a custom-trained YOLOv8 model and ByteTrack multi-object tracking. Runs entirely on-device — no cloud APIs, no accounts, no data leaves your machine.

---

## Problem Statement

Traffic enforcement agencies and smart-city researchers need a way to audit helmet compliance across large volumes of road footage without uploading sensitive imagery to third-party cloud services. This tool provides a self-contained workstation application that runs inference locally, surfaces compliance statistics, and exports structured data for downstream analysis.

**Intended users:** road safety researchers, computer vision students, smart-city analytics teams, and anyone who needs an offline YOLOv8 inference workbench.

---

## Features

### Single Image Analysis
- Drag-and-drop upload or one-click demo image
- Side-by-side original vs. annotated view
- Per-detection table: class, confidence, bounding box coordinates
- Export annotated image (JPEG), CSV, and JSON

### Batch Image Processing
- Multi-file queue — process dozens of images in one run
- Per-image progress bar with inference latency
- Consolidated results table across all images
- One-click ZIP export containing all annotated images plus `batch_summary.csv` and `batch_summary.json`

### Video Analysis (ByteTrack + ROI)
- Supports MP4, AVI, MOV, MKV up to 100 MB
- Region of Interest filtering: Full Frame, Left Lane, Right Lane, Center Lane, or custom normalized coordinates
- ByteTrack multi-object tracking assigns persistent IDs per rider, eliminating duplicate violation counts across frames
- Configurable frame stride (1–5) to trade speed vs. coverage on CPU
- Timeline chart: compliance vs. violations over elapsed video time
- Violation snapshots saved automatically to `violations/`
- Export timeline data as CSV and JSON

### Live Camera Stream
- Real-time webcam inference via `streamlit-webrtc` (WebRTC)
- Thread-safe event queue feeds the live statistics panel without full-page reloads
- Adjustable confidence threshold applied per frame

### Safety Analytics Dashboard
- Cumulative counters: total detections, helmets, violations, compliance rate, unique tracked IDs
- Class distribution bar chart and model confidence histogram (Altair / Vega-Lite)
- Violation archive gallery — review and delete saved snapshots

---

## Technology Stack

| Component | Library / Version |
|---|---|
| UI framework | Streamlit 1.66.0 |
| Object detection | Ultralytics YOLOv8 8.4.182 |
| Multi-object tracking | ByteTrack (via Ultralytics) |
| Computer vision | OpenCV 5.0.0.93 |
| Live camera streaming | streamlit-webrtc 0.78.1 |
| Image processing | Pillow 12.3.0 |
| Video I/O | PyAV 17.1.0 |
| Data / charting | pandas 3.0.6, Altair 6.3.0 |
| Numerical | NumPy 2.5.4 |
| Tracking solver | lap ≥ 0.5.12 |
| Language | Python 3.12 |

---

## Architecture

```
app.py                          ← Streamlit entry point; session state, utility fns
│
├── ui/theme.py                 ← Design token system (light/dark); CSS injection
├── ui/styles.css               ← Central stylesheet (1 000+ lines, CSS custom properties)
├── ui/contrast_patch.css       ← Dark-mode accessibility overrides
├── ui/components.py            ← Reusable HTML render helpers (stat cards, badges)
├── ui/layout.py                ← Sidebar controls, live status fragment, footer
├── ui/icons.py                 ← Inline SVG icon library
│
├── ui/pages/image.py           ← Single Image tab
├── ui/pages/batch.py           ← Batch Processing tab
├── ui/pages/video.py           ← Video Analysis tab (ByteTrack + ROI)
├── ui/pages/live_camera.py     ← Live Camera tab (WebRTC)
└── ui/pages/analytics.py       ← Safety Analytics tab
```

**Data flow (video analysis):**
1. `cv2.VideoCapture` decodes frames at the configured stride.
2. Each frame is passed to `model.track()` (ByteTrack) or `model.predict()`.
3. Detections are filtered by ROI centroid test, then drawn with `results.plot()`.
4. Violation frames (no-helmet class) are snapshotted to `violations/` at most once per second.
5. Per-frame statistics are accumulated in `st.session_state` and rendered live.

---

## Project Structure

```
helmet-detection/
├── .streamlit/
│   └── config.toml             # Streamlit page and theme configuration
├── models/
│   └── best.pt                 # Trained YOLOv8 weights (5.9 MB)
├── sample_media/
│   ├── sample_street.jpg       # Demo image (roadway intersection, 137 KB)
│   └── sample_traffic.mp4      # Demo video (multi-lane traffic, 184 KB)
├── tests/
│   ├── test_pipeline.py        # Automated pipeline tests (6 test cases)
│   └── test_ui_playwright.py   # Headless browser UI tests (Playwright)
├── ui/
│   ├── __init__.py
│   ├── components.py
│   ├── contrast_patch.css
│   ├── icons.py
│   ├── layout.py
│   ├── pages/
│   │   ├── analytics.py
│   │   ├── batch.py
│   │   ├── image.py
│   │   ├── live_camera.py
│   │   └── video.py
│   ├── styles.css
│   └── theme.py
├── violations/                 # Violation snapshots saved here at runtime (gitignored)
├── app.py                      # Main entry point
├── packages.txt                # Linux system dependencies (for cloud deployments)
├── requirements.txt            # Pinned Python dependencies
├── packages.txt                # Linux system dependencies (libgl1, libglib2.0-0t64)
├── run.bat                     # Windows one-click launcher
├── start.sh                    # Linux production startup script
└── README.md
```

---

## Installation (Windows, Python 3.12)

**Prerequisites:** Python 3.12 installed and on your PATH. Git optional.

```powershell
# Clone or download and enter the project directory
cd "D:\Projects\Helmet Detection System\helmet-detection"

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

> On first run, Ultralytics downloads a small ByteTrack configuration file (~2 KB). An internet connection is needed once for this step; subsequent runs are fully offline.

---

## Running the Application

### Option 1 — Windows one-click launcher

```cmd
run.bat
```

The batch file creates `.venv` if absent, installs dependencies, checks for `best.pt`, then starts Streamlit.

### Option 2 — Manual

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Testing

The test suite uses Python's built-in `unittest` — no additional install needed.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_pipeline -v
```

Expected output (all 6 tests pass in ~3 seconds):

```
test_batch_zip_archiving ... ok
test_bytetrack_tracking ... ok
test_file_validation_safeguards ... ok
test_image_inference ... ok
test_model_classes ... ok
test_roi_filtering ... ok

----------------------------------------------------------------------
Ran 6 tests in 2.689s

OK
```

**Test coverage:**
- Model weights load and expose 4 correct class labels
- Single-image inference returns bounding boxes with confidence scores
- ByteTrack assigns persistent integer IDs across sequential frames
- Batch ZIP archive is well-formed and contains expected files
- File validation rejects oversized and wrong-format uploads
- ROI centroid filtering includes/excludes detections correctly

A separate Playwright UI test (`tests/test_ui_playwright.py`) exercises the browser against a running Streamlit server and requires Chrome and Playwright installed separately.

---

## Model

| Property | Value |
|---|---|
| Architecture | YOLOv8 (Ultralytics) |
| Weights file | `runs/train/helmet_detector2/weights/best.pt` |
| File size | 5.9 MB |
| Classes | 4 (see below) |
| Input | RGB images or video frames |
| Inference device | CPU (GPU supported if CUDA available) |

### Class Labels

| ID | Label | Compliance status |
|---|---|---|
| 0 | `bicyclist` | Informational |
| 1 | `driver` | Informational |
| 2 | `helmet` | Compliant |
| 3 | `no-helmet` | Violation |

---

## Supported Formats and Limits

| Mode | Accepted formats | Max size |
|---|---|---|
| Single image | JPG, JPEG, PNG, WEBP, BMP | 15 MB |
| Batch images | JPG, JPEG, PNG, WEBP, BMP | 15 MB per file |
| Video | MP4, AVI, MOV, MKV | 100 MB |
| Live camera | Browser webcam (WebRTC) | — |

---

## Known Limitations

- **No annotated video download** — the video tab exports CSV/JSON timeline data only; saving a re-encoded annotated video is not yet implemented.
- **Live camera requires a physical webcam** — the WebRTC component renders non-functional in headless or server environments; no fallback message is shown.
- **Violation archive is session-global** — files saved during one session persist on disk and appear in subsequent sessions. Use the "Delete Files" button in the sidebar to clear them.
- **CPU inference only by default** — GPU acceleration works if `torch` detects CUDA, but the sidebar device selector currently offers CPU options only.
- **Compliance rate excludes bicyclists and drivers** — the compliance calculation is `helmets / (helmets + no-helmets)`; bicyclist and driver detections are counted in total detections but not in the compliance ratio.

---

## Privacy and File Retention

- All inference runs locally. No images, video frames, or telemetry leave the host machine.
- Violation snapshots are written to `violations/` on disk and persist across application restarts until manually deleted.
- No user accounts, authentication, or cookies.
- Not intended for automated legal enforcement or judicial proceedings.

---

## Deployment

The application is designed for local workstation use. For server deployment:

- `packages.txt` lists the Linux system libraries needed (`libgl1`, `libglib2.0-0t64`) for environments like Streamlit Community Cloud.
- The `violations/` directory must be writable at runtime.
- Live camera (WebRTC) requires STUN/TURN server configuration for non-localhost deployments — see the [streamlit-webrtc documentation](https://github.com/whitphx/streamlit-webrtc).
- `best.pt` (5.9 MB) is committed to the repository and is below GitHub's file size limits. For larger model variants, consider Git LFS.

---

## License

This project uses [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), which is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. As a derivative work, this project is also released under the **AGPL-3.0**.

See [LICENSE](LICENSE) for the full license text.

**Third-party components:**

| Component | License |
|---|---|
| Ultralytics YOLOv8 | AGPL-3.0 |
| Streamlit | Apache 2.0 |
| OpenCV | Apache 2.0 |
| streamlit-webrtc | MIT |
| PyAV | BSD |
| Altair / Vega-Lite | BSD |

> This software is developed for academic research, smart-city safety analytics, and computer vision demonstrations. It is not an automated judicial enforcement or ticketing system.
