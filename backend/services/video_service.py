"""
Video Service
Handles video file management, metadata extraction, and storage.

TODO: Add real video processing (frame extraction, resolution detection, fps)
      using libraries like OpenCV or ffmpeg when integrating real models.
"""

import os
import uuid
import shutil
from datetime import datetime
from fastapi import UploadFile
from models.schemas import VideoMetadata

UPLOAD_DIR = "uploads"


async def save_video(file: UploadFile) -> VideoMetadata:
    """
    Save uploaded video to disk and return metadata.

    TODO: Extract real metadata (duration, resolution, fps) using:
          import cv2
          cap = cv2.VideoCapture(file_path)
          fps = cap.get(cv2.CAP_PROP_FPS)
          frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
          duration = frame_count / fps
    """
    video_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1]
    filename = f"{video_id}{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    # Save file to disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_size = os.path.getsize(file_path)

    metadata = VideoMetadata(
        video_id=video_id,
        filename=file.filename,
        file_size=file_size,
        duration_seconds=None,      # TODO: Extract with cv2
        resolution=None,             # TODO: Extract with cv2
        fps=None,                    # TODO: Extract with cv2
        upload_timestamp=datetime.utcnow().isoformat(),
        file_path=file_path,
    )

    return metadata


def get_video_path(video_id: str) -> str | None:
    """Return the file path for a given video_id, or None if not found."""
    for fname in os.listdir(UPLOAD_DIR):
        if fname.startswith(video_id):
            return os.path.join(UPLOAD_DIR, fname)
    return None


def video_exists(video_id: str) -> bool:
    """Check whether a video with the given ID has been uploaded."""
    return get_video_path(video_id) is not None
