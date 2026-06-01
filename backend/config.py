"""
Central configuration for the Football Match Analysis backend.
All paths and model settings live here — edit this file to switch
from placeholder mode to real model mode.
"""

import os
from pathlib import Path

# ── Base directories ──────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent          # backend/
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR  = BASE_DIR / "models_weights"             # put .pt files here

# Ensure directories exist at import time
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ── Model weight paths ────────────────────────────────────────────────────────
# TODO: Place your trained YOLOv8 .pt weight files in backend/models_weights/
#       and update the filenames below to match.
#
#  Recommended layout:
#    backend/models_weights/
#      player_ball_detector.pt   ← YOLOv8 trained on football player+ball dataset
#      team_classifier.pt        ← optional separate model for team colours
#
MODEL_WEIGHTS_PATH = os.environ.get(
    "MODEL_WEIGHTS_PATH",
    str(MODELS_DIR / "player_ball_detector.pt"),
)

# Class indices as used by your YOLO model.
# Adjust these if your model uses different class IDs.
CLASS_PLAYER    = int(os.environ.get("CLASS_PLAYER", 0))    # "player"
CLASS_BALL      = int(os.environ.get("CLASS_BALL",   1))    # "ball"
CLASS_REFEREE   = int(os.environ.get("CLASS_REFEREE", 2))   # "referee" (optional)
CLASS_GOALKEEPER = int(os.environ.get("CLASS_GK",    3))    # "goalkeeper" (optional)

# ── Inference settings ────────────────────────────────────────────────────────
YOLO_CONF_THRESHOLD = float(os.environ.get("YOLO_CONF", 0.35))
YOLO_IOU_THRESHOLD  = float(os.environ.get("YOLO_IOU",  0.45))
YOLO_IMG_SIZE       = int(os.environ.get("YOLO_IMGSZ", 1280))

# Frame stride: process every Nth frame (1 = every frame, 2 = every other, etc.)
FRAME_STRIDE = int(os.environ.get("FRAME_STRIDE", 2))

# ── ByteTrack settings ────────────────────────────────────────────────────────
BYTETRACK_TRACK_THRESH  = float(os.environ.get("BT_TRACK_THRESH", 0.25))
BYTETRACK_TRACK_BUFFER  = int(os.environ.get("BT_TRACK_BUFFER",  30))
BYTETRACK_MATCH_THRESH  = float(os.environ.get("BT_MATCH_THRESH", 0.8))
BYTETRACK_FRAME_RATE    = int(os.environ.get("BT_FRAME_RATE",    25))

# ── Heatmap settings ─────────────────────────────────────────────────────────
HEATMAP_ALPHA      = float(os.environ.get("HEATMAP_ALPHA", 0.6))
HEATMAP_COLORMAP   = os.environ.get("HEATMAP_COLORMAP", "JET")   # OpenCV colormap name
HEATMAP_GRID_ROWS  = int(os.environ.get("HEATMAP_ROWS", 10))
HEATMAP_GRID_COLS  = int(os.environ.get("HEATMAP_COLS", 16))

# ── Output settings ───────────────────────────────────────────────────────────
OUTPUT_VIDEO_CODEC = os.environ.get("OUTPUT_CODEC", "mp4v")
OUTPUT_VIDEO_EXT   = os.environ.get("OUTPUT_EXT",   ".mp4")

# ── External API (match stats) ────────────────────────────────────────────────
FOOTBALL_API_KEY = os.environ.get("FOOTBALL_API_KEY", "")

# ── Chatbot ───────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
