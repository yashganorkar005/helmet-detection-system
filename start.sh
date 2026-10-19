#!/bin/sh
# Linux launcher for Helmet Detection & Road Safety Analyzer
echo "========================================================"
echo "Starting Helmet Detection & Road Safety Analyzer"
echo "Streamlit + YOLOv8 Deep Learning + ByteTrack"
echo "========================================================"

# Activate virtual environment
if [ -d ".venv" ]; then
    . .venv/bin/activate
else
    echo "[ERROR] Virtual environment not found. Please create it and install requirements."
    exit 1
fi

# Model check
if [ ! -f "models/best.pt" ]; then
    echo "[ERROR] Model checkpoint not found at models/best.pt!"
    exit 1
fi

# Launch application
streamlit run app.py
