"""
cv_detection_service.py
Top-level facade for the CV detection pipeline.
"""

import logging
from pathlib import Path
from typing import Any

from config import MODEL_WEIGHTS_PATH, RESULTS_DIR
from services.cv.video_processor import process_match_video, ProcessingOutput
from models.schemas import DetectionResult, AnalysisStatus

log = logging.getLogger(__name__)


def detect_players_and_ball(video_id: str, video_path: str) -> DetectionResult:
    result_dir = Path(RESULTS_DIR) / video_id
    result_dir.mkdir(parents=True, exist_ok=True)

    output_path = result_dir / "processed.mp4"

    output: ProcessingOutput = process_match_video(
        video_path=str(video_path),
        model_path=str(MODEL_WEIGHTS_PATH),
        output_path=str(output_path),
    )

    data: dict[str, Any] = output.result.get("real_cv_analysis", {})
    status: dict[str, Any] = output.result.get("pipeline_status", {})

    result = DetectionResult(
        video_id=video_id,
        total_frames_analyzed=data.get("frames_processed", 0),
        players_detected=data.get("unique_tracks", 0),
        ball_detections=data.get("total_detections", 0),
        players=[],
        balls=[],
        status=AnalysisStatus.COMPLETED,
    )

    result._cv_output = output  # type: ignore[attr-defined]

    errors = status.get("errors", [])
    if errors:
        log.warning("[cv_detection_service] Pipeline errors: %s", errors)

    return result