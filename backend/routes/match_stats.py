"""
Match statistics routes.

GET  /api/match-stats/video/{video_id}         → stats from CV tracking data
DELETE /api/match-stats/video/{video_id}/cache → evict cached result
GET  /api/match-stats/{match_id}               → legacy alias (treated as video_id)
"""

from fastapi import APIRouter

from models.schemas import MatchStatsResult
from services import match_stats_service

router = APIRouter()


@router.get("/video/{video_id}", response_model=MatchStatsResult)
async def get_match_stats_from_video(video_id: str) -> MatchStatsResult:
    """
    Compute match statistics from CV tracking data for a processed video.

    Loads tracking JSON written by the CV pipeline, runs the full stats
    engine (possession, passing network, sprints, attacks, ball speed, …),
    and returns structured analytics ready for frontend visualisation.

    Returns
    -------
    * ``status=completed`` with populated metrics when tracking data exists.
    * ``status=failed``    when no tracking data is found for *video_id*.

    The result is cached in memory after the first computation.  POST to
    ``/api/analysis/start/{video_id}`` to re-run the pipeline, then call
    ``DELETE /api/match-stats/video/{video_id}/cache`` to clear the stale
    cached result before fetching updated stats.
    """
    return await match_stats_service.fetch_match_stats(video_id=video_id)


@router.delete("/video/{video_id}/cache")
async def clear_stats_cache(video_id: str) -> dict:
    """
    Evict the cached stats result for *video_id*.

    The next GET request will recompute stats from the latest tracking data
    on disk.  Useful after re-running the CV pipeline on the same video.
    """
    match_stats_service.clear_cache(video_id)
    return {"success": True, "message": f"Stats cache cleared for video_id={video_id!r}"}


@router.get("/{match_id}", response_model=MatchStatsResult)
async def get_match_stats(match_id: str) -> MatchStatsResult:
    """
    Legacy endpoint: fetch stats by match_id (treated as video_id).

    Kept for backward compatibility.  Prefer
    ``GET /api/match-stats/video/{video_id}`` for new integrations.
    """
    return await match_stats_service.fetch_match_stats(
        video_id=match_id, match_id=match_id
    )
