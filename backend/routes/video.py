"""
Video upload routes.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from models.schemas import VideoUploadResponse
from services import video_service

router = APIRouter()

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_FILE_SIZE_MB = 500


@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a football match video file.
    Returns a video_id to use in subsequent analysis requests.
    """
    import os

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    try:
        metadata = await video_service.save_video(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")

    return VideoUploadResponse(
        success=True,
        video_id=metadata.video_id,
        metadata=metadata,
        message="Video uploaded successfully. Use video_id to start analysis.",
    )
