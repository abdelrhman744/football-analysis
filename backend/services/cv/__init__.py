"""
CV sub-package.
Exports a single flag `CV_MODEL_AVAILABLE` so the rest of the backend
can decide between real inference and placeholder mode without try/except
scattered everywhere.
"""

import os
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Lazy import guard — ultralytics / cv2 may not be installed yet
try:
    from ultralytics import YOLO  # noqa: F401
    import cv2                    # noqa: F401
    import supervision as sv      # noqa: F401
    _LIBS_AVAILABLE = True
except ImportError as e:
    log.warning("CV libraries not installed (%s). Running in placeholder mode.", e)
    _LIBS_AVAILABLE = False

# Check that the weight file actually exists
from config import MODEL_WEIGHTS_PATH  # noqa: E402

_WEIGHTS_EXIST = Path(MODEL_WEIGHTS_PATH).exists()

if _LIBS_AVAILABLE and not _WEIGHTS_EXIST:
    log.warning(
        "CV libraries are installed but model weights not found at '%s'. "
        "Running in placeholder mode. "
        "Place your YOLOv8 .pt file there and restart the server.",
        MODEL_WEIGHTS_PATH,
    )

CV_MODEL_AVAILABLE: bool = _LIBS_AVAILABLE and _WEIGHTS_EXIST
