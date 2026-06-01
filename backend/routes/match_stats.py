"""
Match statistics routes.
"""

from fastapi import APIRouter
from models.schemas import MatchStatsResult
from services import match_stats_service

router = APIRouter()


@router.get("/{match_id}", response_model=MatchStatsResult)
async def get_match_stats(match_id: str):
    """
    Fetch match statistics for a given match_id from an external football API.
    Currently returns placeholder data.

    TODO: Connect to a real football data API (API-Football, football-data.org, etc.)
          Set the FOOTBALL_API_KEY environment variable and update match_stats_service.py.
    """
    result = await match_stats_service.fetch_match_stats(
        video_id="external", match_id=match_id
    )
    return result
