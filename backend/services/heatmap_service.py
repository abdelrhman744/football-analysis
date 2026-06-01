"""
heatmap_service.py
==================
Top-level facade for heatmap results.

Reads the heatmap data produced by services/cv/heatmap.py (via video_processor)
and converts it into the HeatmapData schema the routes expect.
"""

import logging
from pathlib import Path

from models.schemas import HeatmapData, AnalysisStatus

log = logging.getLogger(__name__)


def _generate_placeholder_matrix(rows: int = 10, cols: int = 15) -> list[list[float]]:
    """
    Return a zeroed heatmap matrix.
    Previously imported from services.cv.heatmap (that module does not exist).
    Inlined here to remove the broken import.
    """
    return [[0.0] * cols for _ in range(rows)]


def _path_to_url(path: str | None) -> str | None:
    """
    Convert an absolute/relative filesystem path inside the results/ directory
    to a URL that FastAPI's StaticFiles mount will serve.

    e.g.  results/abc123/processed_heatmap.jpg
          → /results/abc123/processed_heatmap.jpg
    """
    if not path:
        return None
    p = Path(path)
    try:
        rel = p.relative_to(Path("results"))
        return f"/results/{rel}"
    except ValueError:
        # Path is not under results/ — return as-is so the frontend still gets something
        return str(path)


def generate_heatmap(video_id: str, tracking_result) -> HeatmapData:
    """
    Return HeatmapData populated from real CV output (if available)
    or a zeroed placeholder matrix.

    Reads cv_out from tracking_result._cv_output, which analysis.py sets
    before calling this function.
    """
    cv_out = getattr(tracking_result, "_cv_output", None)

    heatmap_matrix = None
    player_hm_path = None
    ball_hm_path   = None

    if cv_out is not None and getattr(cv_out, "used_real_model", False):
        # heatmap_matrix is None in ProcessingOutput (image only, no matrix yet)
        heatmap_matrix = getattr(cv_out, "heatmap_matrix", None)

        # ProcessingOutput uses heatmap_player_path (singular) — not heatmap_player**s**_path
        player_hm_path = getattr(cv_out, "heatmap_player_path", None) \
                      or getattr(cv_out, "heatmap_players_path", None)

        ball_hm_path = getattr(cv_out, "heatmap_ball_path", None)

    if heatmap_matrix is None:
        heatmap_matrix = _generate_placeholder_matrix()

    return HeatmapData(
        video_id=video_id,
        team_a_heatmap_url=_path_to_url(player_hm_path),
        team_b_heatmap_url=None,
        ball_heatmap_url=_path_to_url(ball_hm_path),
        heatmap_matrix=heatmap_matrix,
        status=AnalysisStatus.COMPLETED,
    )
