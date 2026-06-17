"""
heatmap_service.py
==================
Top-level facade for heatmap results.

Reads the heatmap data produced by services/cv/heatmap.py (via video_processor)
and converts it into the HeatmapData schema the routes expect.

Fix: _path_to_url now resolves against RESULTS_DIR as an absolute Path so that
     absolute filesystem paths returned by video_processor are correctly
     converted to /results/... URL paths that FastAPI's StaticFiles can serve.
"""

import logging
from pathlib import Path

from config import RESULTS_DIR
from models.schemas import HeatmapData, AnalysisStatus

log = logging.getLogger(__name__)


def _generate_placeholder_matrix(rows: int = 10, cols: int = 15) -> list[list[float]]:
    """Return a zeroed heatmap matrix used when no real CV data is available."""
    return [[0.0] * cols for _ in range(rows)]


def _path_to_url(path: str | None) -> str | None:
    """
    Convert an absolute or relative filesystem path inside the results/ directory
    to a URL that FastAPI's StaticFiles mount will serve.

    Handles both:
      - Absolute paths:  /abs/path/to/results/abc/heatmap.jpg → /results/abc/heatmap.jpg
      - Relative paths:  results/abc/heatmap.jpg              → /results/abc/heatmap.jpg
    """
    if not path:
        return None

    p = Path(path).resolve()
    results_abs = Path(RESULTS_DIR).resolve()

    try:
        rel = p.relative_to(results_abs)
        return f"/results/{rel}"
    except ValueError:
        pass

    # Fallback: try treating path as relative to current working directory
    p_rel = Path(path)
    try:
        rel = p_rel.relative_to(Path("results"))
        return f"/results/{rel}"
    except ValueError:
        pass

    # Last resort: return the path as-is so the frontend still gets something
    log.warning("[heatmap_service] Cannot convert path to URL: %s", path)
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
        heatmap_matrix = getattr(cv_out, "heatmap_matrix", None)

        # ProcessingOutput uses heatmap_player_path (singular)
        player_hm_path = (
            getattr(cv_out, "heatmap_player_path",  None)
            or getattr(cv_out, "heatmap_players_path", None)
        )
        ball_hm_path = getattr(cv_out, "heatmap_ball_path", None)

        log.debug(
            "[heatmap_service] raw player_hm_path=%s  ball_hm_path=%s",
            player_hm_path, ball_hm_path,
        )

    if heatmap_matrix is None:
        heatmap_matrix = _generate_placeholder_matrix()

    player_url = _path_to_url(player_hm_path)
    ball_url   = _path_to_url(ball_hm_path)

    log.info(
        "[heatmap_service] video_id=%s  player_url=%s  ball_url=%s",
        video_id, player_url, ball_url,
    )

    return HeatmapData(
        video_id=video_id,
        team_a_heatmap_url=player_url,
        team_b_heatmap_url=None,
        ball_heatmap_url=ball_url,
        heatmap_matrix=heatmap_matrix,
        status=AnalysisStatus.COMPLETED,
    )
