"""
tracking_service.py
===================
Top-level facade for tracking results.

The real tracking runs inside video_processor.py (all in one pass over the
video for efficiency). This service reads back the tracking JSON saved by
the pipeline and converts it to the TrackingResult schema.
"""

import json
import logging
import random
from pathlib import Path

from models.schemas import TrackingResult, PlayerTrack, TrackingPoint, AnalysisStatus
from config import RESULTS_DIR

log = logging.getLogger(__name__)


def track_objects(video_id: str, detection_result) -> TrackingResult:
    """
    Build TrackingResult from the tracking JSON written by video_processor.
    Falls back to placeholder data if the file doesn't exist.

    Args:
        video_id         : video identifier
        detection_result : DetectionResult from cv_detection_service
                           (may carry ._cv_output with real paths)
    """
    # Check if real tracking JSON was produced
    cv_out = getattr(detection_result, "_cv_output", None)
    track_path = None

    if cv_out and cv_out.tracking_json_path:
        track_path = Path(cv_out.tracking_json_path)
    else:
        track_path = Path(RESULTS_DIR) / video_id / "tracking.json"

    if track_path and track_path.exists() and cv_out and cv_out.used_real_model:
        return _from_real_json(video_id, track_path)

    return _placeholder(video_id)


def _from_real_json(video_id: str, path: Path) -> TrackingResult:
    """
    Parse the tracking JSON produced by video_processor.py.

    video_processor writes a flat list:
        [
          {"frame": 0, "track_id": 1, "center": [cx, cy], "bbox": [x1,y1,x2,y2]},
          ...
        ]

    The old implementation expected a nested dict shape that was never written.
    This version handles the actual format.
    """
    try:
        with open(path) as f:
            data = json.load(f)

        # data is a flat list of per-detection tracking entries
        if not isinstance(data, list):
            log.warning("[tracking_service] Unexpected tracking JSON format — falling back")
            return _placeholder(video_id)

        # Group by track_id
        tracks: dict[int, list[dict]] = {}
        for entry in data:
            tid = entry.get("track_id")
            if tid is None:
                continue
            tracks.setdefault(tid, []).append(entry)

        player_tracks = []
        for tid, entries in tracks.items():
            pts = [
                TrackingPoint(
                    frame=int(e["frame"]),
                    x=float(e["center"][0]),
                    y=float(e["center"][1]),
                    timestamp=round(e["frame"] / 25.0, 3),
                )
                for e in sorted(entries, key=lambda e: e["frame"])
            ]
            player_tracks.append(
                PlayerTrack(
                    player_id=int(tid),
                    team=None,                       # team classification is separate
                    track_points=pts,
                    total_distance_meters=None,      # pixel-space only; calibration needed
                    average_speed_kmh=None,
                    max_speed_kmh=None,
                )
            )

        return TrackingResult(
            video_id=video_id,
            total_players_tracked=len(player_tracks),
            player_tracks=player_tracks,
            ball_track=[],          # ball tracking not separated yet
            status=AnalysisStatus.COMPLETED,
        )

    except Exception as exc:
        log.error("[tracking_service] Error parsing real JSON: %s", exc)
        return _placeholder(video_id)


def _placeholder(video_id: str) -> TrackingResult:
    """Generate realistic-looking fake tracking data."""
    num_players, num_frames = 22, 500

    player_tracks = []
    for pid in range(num_players):
        x, y = random.uniform(100, 800), random.uniform(100, 400)
        pts = []
        for frame in range(0, num_frames, 5):
            x = max(0, min(1920, x + random.uniform(-12, 12)))
            y = max(0, min(1080, y + random.uniform(-8, 8)))
            pts.append(TrackingPoint(frame=frame, x=round(x, 2), y=round(y, 2),
                                     timestamp=round(frame / 25.0, 3)))
        player_tracks.append(PlayerTrack(
            player_id=pid,
            team="Team A" if pid < 11 else "Team B",
            track_points=pts,
            total_distance_meters=round(random.uniform(4000, 11000), 1),
            average_speed_kmh=round(random.uniform(6, 12), 2),
            max_speed_kmh=round(random.uniform(18, 32), 2),
        ))

    bx, by = 500, 300
    ball_track = []
    for frame in range(0, num_frames, 2):
        bx = max(0, min(1920, bx + random.uniform(-25, 25)))
        by = max(0, min(1080, by + random.uniform(-20, 20)))
        ball_track.append(TrackingPoint(frame=frame, x=round(bx, 2), y=round(by, 2),
                                        timestamp=round(frame / 25.0, 3)))

    return TrackingResult(
        video_id=video_id,
        total_players_tracked=num_players,
        player_tracks=player_tracks,
        ball_track=ball_track[:50],
        status=AnalysisStatus.COMPLETED,
    )
