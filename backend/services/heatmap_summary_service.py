"""
Heatmap Summary Service
========================
Generates human-readable, per-player heatmap summaries from CV tracking data.

This is new code (not part of the three provided stats files) because none of
them produce a text summary — only numeric heatmap matrices. It reuses the
already-normalized tracking entries produced by ``stats_engine.load_tracking``
rather than re-deriving anything, so there is no duplicated parsing logic.

Output is consumed by:
  * The frontend dashboard (human-readable caption under each heatmap)
  * The future Football Analyst Chatbot via RAG
    (persisted to results/{video_id}/heatmap_summaries.json by routes/analysis.py)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

# Pitch split into a 3x3 grid. X runs along the length of the pitch
# (defensive → attacking, from each team's own attacking direction).
# Y runs across the width of the pitch (left flank → right flank).
X_ZONE_NAMES = ["defensive third", "midfield", "attacking third"]
Y_ZONE_NAMES = ["left flank", "central channel", "right flank"]


def generate_player_heatmap_summaries(
    tracking_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build a most-active-zone text summary for every tracked player.

    Args:
        tracking_data: Normalized entries as returned by
            ``services.stats_engine.load_tracking`` (each entry has
            ``track_id``, ``class``, ``team``, ``center``).

    Returns:
        List of summaries, one per player track, e.g.::

            {
              "player_id": 7,
              "team": "A",
              "most_active_zone": "right flank (attacking third)",
              "zone_share_pct": 42.7,
              "summary": "Player #7 (Team A) spent 42.7% of tracked frames ..."
            }
    """
    if not tracking_data:
        return []

    player_points = [e for e in tracking_data if e.get("class") == "player" and "center" in e]
    if not player_points:
        return []

    x_vals = [p["center"][0] for p in player_points]
    y_vals = [p["center"][1] for p in player_points]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    x_range = max(x_max - x_min, 1.0)
    y_range = max(y_max - y_min, 1.0)

    tracks: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    team_by_player: dict[int, str | None] = {}

    for entry in player_points:
        try:
            track_id = int(entry["track_id"])
        except (TypeError, ValueError):
            continue
        tracks[track_id].append(entry)
        if entry.get("team") in ("A", "B"):
            team_by_player[track_id] = entry["team"]

    summaries: list[dict[str, Any]] = []

    for track_id, points in tracks.items():
        team = team_by_player.get(track_id)
        zone_counts: Counter[str] = Counter()

        for point in points:
            cx, cy = point["center"]
            x_idx = min(int((cx - x_min) / x_range * 3), 2)
            y_idx = min(int((cy - y_min) / y_range * 3), 2)
            # Team B attacks in the opposite direction, so mirror its x-axis
            # zone labels (its "attacking third" is the pitch's low-x side).
            x_zone = X_ZONE_NAMES[x_idx if team != "B" else 2 - x_idx]
            zone_name = f"{Y_ZONE_NAMES[y_idx]} ({x_zone})"
            zone_counts[zone_name] += 1

        if not zone_counts:
            continue

        total = sum(zone_counts.values())
        zone, count = zone_counts.most_common(1)[0]
        share_pct = round(count / total * 100.0, 1)
        team_label = f"Team {team}" if team else "an unclassified team"

        summaries.append(
            {
                "player_id": track_id,
                "team": team,
                "most_active_zone": zone,
                "zone_share_pct": share_pct,
                "summary": (
                    f"Player #{track_id} ({team_label}) spent {share_pct}% of tracked "
                    f"frames in the {zone} zone, making it their most active area "
                    f"of the pitch during this match."
                ),
            }
        )

    summaries.sort(key=lambda s: s["player_id"])
    return summaries
