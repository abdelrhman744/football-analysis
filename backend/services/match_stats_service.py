"""
Match Statistics Service
========================
Replaces the old placeholder implementation with real CV-derived analytics.

Architecture
------------
* stats_engine  — pure analytics (no I/O, no API logic)
* THIS file     — loads data, orchestrates computation, formats the response,
                  and owns the in-process cache.

Cache
-----
A simple dict cache keyed by video_id.  Call ``clear_cache(video_id)`` after
re-processing a video so the next request recomputes fresh numbers.

Error handling
--------------
* No tracking data → AnalysisStatus.FAILED (empty TeamStats, no metrics).
* Load error       → AnalysisStatus.FAILED (logged server-side).
* Partial data     → AnalysisStatus.COMPLETED with whatever could be computed
                     (individual metric functions all return safe defaults).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from models.schemas import AnalysisStatus, MatchStatsResult, TeamStats
from services.stats_engine import (
    compute_heatmap_matrix,
    compute_match_stats,
    compute_possession_zones,
    load_tracking,
)

log = logging.getLogger(__name__)

# ── In-memory result cache ────────────────────────────────────────────────────
_stats_cache: dict[str, MatchStatsResult] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def clear_cache(video_id: str | None = None) -> None:
    """
    Evict cache entry for *video_id*, or flush the entire cache if None.

    Call this after re-running the CV pipeline on a video so the next
    stats request picks up fresh tracking data.
    """
    if video_id is None:
        _stats_cache.clear()
        log.info("[match_stats_service] Full cache cleared")
    else:
        evicted = _stats_cache.pop(video_id, None)
        if evicted:
            log.info("[match_stats_service] Cache evicted for video_id=%s", video_id)


async def fetch_match_stats(
    video_id: str,
    match_id: Optional[str] = None,
) -> MatchStatsResult:
    """
    Compute and return match statistics derived from CV tracking data.

    Pipeline
    --------
    1.  Check in-memory cache — return early on hit.
    2.  Load tracking entries from disk via ``stats_engine.load_tracking``.
    3.  Return FAILED result if no entries found.
    4.  Run ``compute_match_stats`` for core metrics.
    5.  Run bonus analytics (possession zones, per-team heatmaps) — each
        degrades gracefully; a failure does **not** abort the response.
    6.  Map everything to ``MatchStatsResult`` and cache before returning.

    Args:
        video_id:  The video identifier used to locate result files on disk.
        match_id:  Optional external identifier stored in the response metadata.

    Returns:
        ``MatchStatsResult`` with ``status=COMPLETED`` on success or
        ``status=FAILED`` when no tracking data is available.
    """
    # ── 1. Cache check ────────────────────────────────────────────────────────
    if video_id in _stats_cache:
        log.info("[match_stats_service] Cache hit for video_id=%s", video_id)
        return _stats_cache[video_id]

    # ── 2. Load tracking data ─────────────────────────────────────────────────
    try:
        tracking_data = load_tracking(video_id)
    except Exception as exc:
        log.error(
            "[match_stats_service] Failed to load tracking for video_id=%s: %s",
            video_id,
            exc,
        )
        return _failed_result(video_id, match_id)

    # ── 3. Guard: nothing to compute ──────────────────────────────────────────
    if not tracking_data:
        log.warning(
            "[match_stats_service] No tracking data for video_id=%s — returning FAILED",
            video_id,
        )
        return _failed_result(video_id, match_id)

    log.info(
        "[match_stats_service] Loaded %d tracking entries for video_id=%s",
        len(tracking_data),
        video_id,
    )

    # ── 4. Core metrics ───────────────────────────────────────────────────────
    try:
        stats = compute_match_stats(tracking_data)
    except Exception as exc:
        log.error(
            "[match_stats_service] Stats computation failed for video_id=%s: %s",
            video_id,
            exc,
        )
        return _failed_result(video_id, match_id)

    # ── 5. Bonus analytics — non-critical, each wrapped independently ─────────
    possession_zones: dict[str, Any] = {}
    heatmap_matrix_a: list[list[float]] = []
    heatmap_matrix_b: list[list[float]] = []

    try:
        possession_zones = compute_possession_zones(tracking_data)
    except Exception as exc:  # pragma: no cover
        log.warning("[match_stats_service] Possession zones failed: %s", exc)

    try:
        heatmap_matrix_a = compute_heatmap_matrix(tracking_data, team="A")
        heatmap_matrix_b = compute_heatmap_matrix(tracking_data, team="B")
    except Exception as exc:  # pragma: no cover
        log.warning("[match_stats_service] Heatmap computation failed: %s", exc)

    # ── 6. Build response ─────────────────────────────────────────────────────
    result = _build_result(
        video_id=video_id,
        match_id=match_id,
        stats=stats,
        tracking_data=tracking_data,
        possession_zones=possession_zones,
        heatmap_matrix_a=heatmap_matrix_a,
        heatmap_matrix_b=heatmap_matrix_b,
    )

    _stats_cache[video_id] = result
    log.info(
        "[match_stats_service] Stats computed and cached for video_id=%s "
        "(possession A=%.1f%% B=%.1f%%)",
        video_id,
        result.possession.get("A", 0.0),
        result.possession.get("B", 0.0),
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers — service-layer mapping only, no analytics logic
# ─────────────────────────────────────────────────────────────────────────────


def _build_result(
    *,
    video_id: str,
    match_id: Optional[str],
    stats: dict[str, Any],
    tracking_data: list[dict[str, Any]],
    possession_zones: dict[str, Any],
    heatmap_matrix_a: list[list[float]],
    heatmap_matrix_b: list[list[float]],
) -> MatchStatsResult:
    """Map the raw stats dict + bonus data into a ``MatchStatsResult``."""
    # Core metrics from compute_match_stats
    possession: dict[str, float] = stats.get("possession", {"A": 0.0, "B": 0.0})
    passing_network: dict[str, int] = stats.get("passes", {})
    ball_recovery: dict[str, int] = stats.get("ball_recovery", {"A": 0, "B": 0})
    attacks: dict[str, int] = stats.get("attacks", {"A": 0, "B": 0})
    sprints: dict[str, int] = stats.get("sprints", {"A": 0, "B": 0})
    ball_speed: dict[str, float] = stats.get("ball_speed", {"avg": 0.0, "max": 0.0})
    per_player: dict[str, Any] = stats.get("per_player", {})

    # Resolve team membership for pass counting
    team_by_player = _build_team_lookup(tracking_data)

    # Per-team pass totals (sum of outgoing edges per team)
    passes_a = _count_team_passes(passing_network, team_by_player, "A")
    passes_b = _count_team_passes(passing_network, team_by_player, "B")

    # Estimate pass accuracy using ball recoveries as a proxy for turnovers:
    # Team A turnovers ≈ times Team B recovered the ball (ball_recovery["B"])
    acc_a = _estimate_pass_accuracy(passes_a, ball_recovery.get("B", 0))
    acc_b = _estimate_pass_accuracy(passes_b, ball_recovery.get("A", 0))

    home_stats = TeamStats(
        team_name="Team A",
        possession_percentage=possession.get("A", 0.0),
        passes=passes_a,
        pass_accuracy=acc_a,
    )
    away_stats = TeamStats(
        team_name="Team B",
        possession_percentage=possession.get("B", 0.0),
        passes=passes_b,
        pass_accuracy=acc_b,
    )

    return MatchStatsResult(
        match_id=match_id,
        video_id=video_id,
        home_team=home_stats,
        away_team=away_stats,
        possession=possession,
        ball_recovery=ball_recovery,
        # passes = per-team totals for quick access
        passes={"A": passes_a, "B": passes_b},
        # passing_network = full edge-level detail for visualisation
        passing_network=passing_network,
        attacks=attacks,
        sprints=sprints,
        ball_speed=ball_speed,
        per_player=per_player,
        # bonus fields
        possession_zones=possession_zones,
        heatmap_matrix_a=heatmap_matrix_a,
        heatmap_matrix_b=heatmap_matrix_b,
        status=AnalysisStatus.COMPLETED,
        data_source="cv_tracking",
    )


def _failed_result(video_id: str, match_id: Optional[str]) -> MatchStatsResult:
    """Minimal response when tracking data is missing or computation fails."""
    return MatchStatsResult(
        match_id=match_id,
        video_id=video_id,
        home_team=TeamStats(team_name="Team A"),
        away_team=TeamStats(team_name="Team B"),
        status=AnalysisStatus.FAILED,
        data_source="cv_tracking",
    )


def _build_team_lookup(tracking_data: list[dict[str, Any]]) -> dict[int, str]:
    """Return ``{track_id: team}`` mapping built from player entries."""
    lookup: dict[int, str] = {}
    for entry in tracking_data:
        if entry.get("class") == "player" and entry.get("team") in ("A", "B"):
            lookup[int(entry["track_id"])] = entry["team"]
    return lookup


def _count_team_passes(
    passing_network: dict[str, int],
    team_by_player: dict[int, str],
    team: str,
) -> int:
    """Sum pass counts for edges whose *source* player belongs to ``team``."""
    total = 0
    for edge, count in passing_network.items():
        try:
            src_id = int(edge.split("->")[0])
        except (ValueError, IndexError):
            continue
        if team_by_player.get(src_id) == team:
            total += count
    return total


def _estimate_pass_accuracy(
    passes_completed: int,
    turnovers: int,
) -> Optional[float]:
    """
    Estimate pass accuracy from completed passes and turnover proxy.

    Formula:  accuracy = passes / (passes + turnovers) × 100

    Returns None if there is insufficient data to form a meaningful estimate
    (i.e. zero total attempts).
    """
    total = passes_completed + turnovers
    if total == 0:
        return None
    return round(passes_completed / total * 100.0, 1)
