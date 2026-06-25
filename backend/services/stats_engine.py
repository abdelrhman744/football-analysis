"""
Match statistics engine derived entirely from CV tracking data.

The expected tracking shape is a flat list of entries like:
    {
        "frame": int,
        "track_id": int,
        "class": "player" | "ball",
        "team": "A" | "B",
        "center": [x, y],
    }

The loader also accepts the sidecar files produced by the current CV pipeline
(`processed_tracking.json`, `processed_detections.json`, and
`processed_team_assignments.json`) and enriches entries when class/team fields
are missing.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from config import RESULTS_DIR

log = logging.getLogger(__name__)

FPS = 25
PIXEL_TO_METER = 0.05

POSSESSION_MAX_DISTANCE_PX = 120.0
PASS_MAX_GAP_FRAMES = FPS * 3
ATTACK_SPEED_THRESHOLD_MPS = 3.0
SPRINT_SPEED_THRESHOLD_MPS = 7.0
BALL_SMOOTHING_WINDOW = 3
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


def load_tracking(video_id: str) -> list[dict[str, Any]]:
    """Load and normalize tracking data for a video_id."""
    result_dir = Path(RESULTS_DIR) / video_id
    tracking_path = _find_first_existing(
        result_dir / "tracking.json",
        result_dir / "processed_tracking.json",
        *sorted(result_dir.glob("*_tracking.json")),
    )
    if tracking_path is None:
        log.warning("[stats_engine] No tracking data found for video_id=%s", video_id)
        return []

    try:
        with open(tracking_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as exc:
        log.error("[stats_engine] Could not read tracking file %s: %s", tracking_path, exc)
        return []

    if not isinstance(raw_data, list):
        log.warning("[stats_engine] Tracking file has unexpected shape: %s", tracking_path)
        return []

    detections_by_key = _load_detection_index(result_dir)
    team_by_track = _load_team_assignments(result_dir)

    normalized: list[dict[str, Any]] = []
    for raw in raw_data:
        entry = _normalize_entry(raw, detections_by_key, team_by_track)
        if entry is not None:
            normalized.append(entry)

    normalized.sort(key=lambda item: (item["frame"], item["class"] != "ball", item["track_id"]))
    return normalized


def compute_possession(
    tracking_data: list[dict[str, Any]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """
    Assign each frame's ball possession to the team of the nearest player.

    Frames without a ball, without players, or with an implausibly distant
    nearest player are ignored rather than forcing noisy possession.
    """
    frames = _group_by_frame(tracking_data)
    counts: Counter[str] = Counter()
    timeline: list[dict[str, Any]] = []

    for frame, entries in frames.items():
        ball = _select_ball(entries)
        players = [e for e in entries if e.get("class") == "player" and e.get("team") in ("A", "B")]
        if ball is None or not players:
            continue

        ball_center = np.asarray(ball["center"], dtype=float)
        player_centers = np.asarray([p["center"] for p in players], dtype=float)
        distances = np.linalg.norm(player_centers - ball_center, axis=1)
        nearest_idx = int(np.argmin(distances))
        nearest_distance = float(distances[nearest_idx])
        if nearest_distance > POSSESSION_MAX_DISTANCE_PX:
            continue

        owner = players[nearest_idx]
        team = owner["team"]
        counts[team] += 1
        timeline.append(
            {
                "frame": frame,
                "team": team,
                "owner": owner["track_id"],
                "distance_px": nearest_distance,
                "ball_center": ball["center"],
            }
        )

    total = counts["A"] + counts["B"]
    if total == 0:
        return {"A": 0.0, "B": 0.0}, timeline

    possession = {
        "A": round((counts["A"] / total) * 100.0, 1),
        "B": round((counts["B"] / total) * 100.0, 1),
    }
    # Keep the visible numbers summing to exactly 100.0 after rounding.
    if total > 0:
        possession["B"] = round(100.0 - possession["A"], 1)
    return possession, timeline


def compute_ball_recovery(possession_timeline: list[dict[str, Any]]) -> dict[str, int]:
    """Count recoveries when possession changes directly from one team to the other."""
    recoveries = {"A": 0, "B": 0}
    previous_team: str | None = None

    for point in possession_timeline:
        team = point.get("team")
        if team not in recoveries:
            continue
        if previous_team is not None and team != previous_team:
            recoveries[team] += 1
        previous_team = team

    return recoveries


def compute_passing_network(tracking_data: list[dict[str, Any]]) -> dict[str, int]:
    """Build a player-to-player pass network from same-team owner changes."""
    _, timeline = compute_possession(tracking_data)
    passes: defaultdict[tuple[int, int], int] = defaultdict(int)
    previous: dict[str, Any] | None = None

    for point in timeline:
        if previous is None:
            previous = point
            continue

        same_team = point["team"] == previous["team"]
        new_owner = point["owner"] != previous["owner"]
        close_enough = (point["frame"] - previous["frame"]) <= PASS_MAX_GAP_FRAMES
        if same_team and new_owner and close_enough:
            passes[(int(previous["owner"]), int(point["owner"]))] += 1

        previous = point

    return {f"{src}->{dst}": count for (src, dst), count in sorted(passes.items())}


def compute_attacks(tracking_data: list[dict[str, Any]]) -> dict[str, int]:
    """
    Count attacking sequences.

    Team A is assumed to attack left-to-right, Team B right-to-left. An attack
    starts when the possessed ball is moving toward the opponent goal above the
    configured speed threshold; continuous qualifying frames count as one attack.
    """
    _, possession_timeline = compute_possession(tracking_data)
    owner_by_frame = {point["frame"]: point for point in possession_timeline}
    ball_points = _smoothed_ball_points(tracking_data)
    attacks = {"A": 0, "B": 0}
    active_team: str | None = None

    for prev, cur in zip(ball_points, ball_points[1:]):
        frame_gap = max(int(cur["frame"]) - int(prev["frame"]), 1)
        dt = frame_gap / FPS
        dx = float(cur["center"][0]) - float(prev["center"][0])
        distance_m = _distance_m(prev["center"], cur["center"])
        speed_mps = distance_m / dt if dt > 0 else 0.0

        owner = owner_by_frame.get(cur["frame"])
        team = owner.get("team") if owner else None
        moving_forward = (team == "A" and dx > 0) or (team == "B" and dx < 0)
        is_attack = bool(team in attacks and moving_forward and speed_mps >= ATTACK_SPEED_THRESHOLD_MPS)

        if is_attack:
            if active_team != team:
                attacks[team] += 1
                active_team = team
        else:
            active_team = None

    return attacks


def compute_sprints(tracking_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Count sprint events per team and per player from frame-to-frame speed."""
    tracks: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    team_by_player: dict[int, str] = {}

    for entry in tracking_data:
        if entry.get("class") != "player":
            continue
        track_id = int(entry["track_id"])
        tracks[track_id].append(entry)
        if entry.get("team") in ("A", "B"):
            team_by_player[track_id] = entry["team"]

    team_counts = {"A": 0, "B": 0}
    per_player: dict[str, dict[str, Any]] = {}

    for track_id, points in tracks.items():
        points.sort(key=lambda item: item["frame"])
        sprint_count = 0
        in_sprint = False
        max_speed = 0.0

        for prev, cur in zip(points, points[1:]):
            frame_gap = int(cur["frame"]) - int(prev["frame"])
            if frame_gap <= 0:
                continue
            speed_mps = _distance_m(prev["center"], cur["center"]) / (frame_gap / FPS)
            max_speed = max(max_speed, speed_mps)
            if speed_mps > SPRINT_SPEED_THRESHOLD_MPS:
                if not in_sprint:
                    sprint_count += 1
                    in_sprint = True
            else:
                in_sprint = False

        team = team_by_player.get(track_id)
        if team in team_counts:
            team_counts[team] += sprint_count

        per_player[str(track_id)] = {
            "team": team,
            "sprints": sprint_count,
            "max_speed": round(max_speed, 2),
        }

    return {"teams": team_counts, "players": per_player}


def compute_ball_speed(tracking_data: list[dict[str, Any]]) -> dict[str, float]:
    """Compute average and max ball speed in meters per second."""
    ball_points = _smoothed_ball_points(tracking_data)
    if len(ball_points) < 2:
        return {"avg": 0.0, "max": 0.0}

    speeds: list[float] = []
    for prev, cur in zip(ball_points, ball_points[1:]):
        frame_gap = int(cur["frame"]) - int(prev["frame"])
        if frame_gap <= 0:
            continue
        speeds.append(_distance_m(prev["center"], cur["center"]) / (frame_gap / FPS))

    if not speeds:
        return {"avg": 0.0, "max": 0.0}
    return {"avg": round(float(np.mean(speeds)), 2), "max": round(float(np.max(speeds)), 2)}


def compute_match_stats(tracking_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the complete match statistics payload."""
    possession, timeline = compute_possession(tracking_data)
    sprints = compute_sprints(tracking_data)
    per_player = _build_per_player_stats(tracking_data, sprints["players"])

    return {
        "possession": possession,
        "ball_recovery": compute_ball_recovery(timeline),
        "passes": compute_passing_network(tracking_data),
        "attacks": compute_attacks(tracking_data),
        "sprints": sprints["teams"],
        "ball_speed": compute_ball_speed(tracking_data),
        "per_player": per_player,
    }


def _find_first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _load_detection_index(result_dir: Path) -> dict[tuple[int, int], dict[str, Any]]:
    detection_path = _find_first_existing(
        result_dir / "processed_detections.json",
        *sorted(result_dir.glob("*_detections.json")),
    )
    if detection_path is None:
        return {}

    try:
        with open(detection_path, "r", encoding="utf-8") as f:
            frames = json.load(f)
    except Exception:
        return {}

    index: dict[tuple[int, int], dict[str, Any]] = {}
    if not isinstance(frames, list):
        return index

    for frame_payload in frames:
        frame = frame_payload.get("frame")
        for det in frame_payload.get("detections", []):
            tracker_id = det.get("tracker_id")
            if frame is not None and tracker_id is not None:
                index[(int(frame), int(tracker_id))] = det
    return index


def _load_team_assignments(result_dir: Path) -> dict[int, str]:
    assignment_path = _find_first_existing(
        result_dir / "processed_team_assignments.json",
        *sorted(result_dir.glob("*_team_assignments.json")),
    )
    if assignment_path is None:
        return {}

    try:
        with open(assignment_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}

    assignments = payload.get("assignments", payload) if isinstance(payload, dict) else {}
    team_by_track: dict[int, str] = {}
    for raw_track_id, raw_team in assignments.items():
        team = _normalize_team(raw_team)
        if team in ("A", "B"):
            team_by_track[int(raw_track_id)] = team
    return team_by_track


def _normalize_entry(
    raw: dict[str, Any],
    detections_by_key: dict[tuple[int, int], dict[str, Any]],
    team_by_track: dict[int, str],
) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or "center" not in raw:
        return None

    try:
        frame = int(raw.get("frame", 0))
        track_id = int(raw.get("track_id", raw.get("tracker_id", -1)))
        center = [float(raw["center"][0]), float(raw["center"][1])]
    except (TypeError, ValueError, IndexError):
        return None

    det = detections_by_key.get((frame, track_id), {})
    class_name = _normalize_class(raw.get("class"), raw.get("class_id", det.get("class_id")))
    team = _normalize_team(raw.get("team", team_by_track.get(track_id)))

    return {
        "frame": frame,
        "track_id": track_id,
        "class": class_name,
        "team": team,
        "center": center,
    }


def _normalize_class(raw_class: Any, raw_class_id: Any = None) -> str:
    if isinstance(raw_class, str):
        value = raw_class.lower()
        if value in {"ball", "football"}:
            return "ball"
        if value in {"player", "goalkeeper"}:
            return "player"
    try:
        return "ball" if int(raw_class_id) == 0 else "player"
    except (TypeError, ValueError):
        return "player"


def _normalize_team(raw_team: Any) -> str | None:
    if raw_team is None:
        return None
    if isinstance(raw_team, str):
        value = raw_team.strip().upper()
        if value in {"A", "TEAM A", "TEAM 1", "1"}:
            return "A"
        if value in {"B", "TEAM B", "TEAM 2", "2"}:
            return "B"
    try:
        numeric = int(raw_team)
    except (TypeError, ValueError):
        return None
    return "A" if numeric == 1 else "B" if numeric == 2 else None


def _group_by_frame(tracking_data: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    frames: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in tracking_data:
        frames[int(entry["frame"])].append(entry)
    return dict(sorted(frames.items()))


def _select_ball(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    balls = [entry for entry in entries if entry.get("class") == "ball"]
    if not balls:
        return None
    return balls[0]


def _smoothed_ball_points(tracking_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ball_points = sorted(
        (entry for entry in tracking_data if entry.get("class") == "ball"),
        key=lambda item: item["frame"],
    )
    if len(ball_points) <= BALL_SMOOTHING_WINDOW:
        return ball_points

    smoothed: list[dict[str, Any]] = []
    half_window = BALL_SMOOTHING_WINDOW // 2
    centers = np.asarray([point["center"] for point in ball_points], dtype=float)

    for idx, point in enumerate(ball_points):
        start = max(0, idx - half_window)
        end = min(len(ball_points), idx + half_window + 1)
        center = np.mean(centers[start:end], axis=0)
        smoothed.append({**point, "center": [float(center[0]), float(center[1])]})

    return smoothed


def _distance_m(a: list[float], b: list[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))) * PIXEL_TO_METER


def _build_per_player_stats(
    tracking_data: list[dict[str, Any]],
    sprint_players: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    tracks: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    max_x = max((entry["center"][0] for entry in tracking_data), default=1.0)
    max_y = max((entry["center"][1] for entry in tracking_data), default=1.0)

    for entry in tracking_data:
        if entry.get("class") == "player":
            tracks[int(entry["track_id"])].append(entry)

    per_player: dict[str, dict[str, Any]] = {}
    for track_id, points in tracks.items():
        points.sort(key=lambda item: item["frame"])
        distance = sum(_distance_m(prev["center"], cur["center"]) for prev, cur in zip(points, points[1:]))
        last = points[-1] if points else None
        player_key = str(track_id)
        per_player[player_key] = {
            "team": sprint_players.get(player_key, {}).get("team"),
            "distance_m": round(distance, 2),
            "sprints": sprint_players.get(player_key, {}).get("sprints", 0),
            "max_speed": sprint_players.get(player_key, {}).get("max_speed", 0.0),
            "normalized_position": _normalize_pitch_position(last["center"], max_x, max_y) if last else None,
        }
    return per_player


def _normalize_pitch_position(center: list[float], max_x: float, max_y: float) -> dict[str, float]:
    width = max(max_x, 1.0)
    height = max(max_y, 1.0)
    return {
        "x_m": round((float(center[0]) / width) * PITCH_LENGTH_M, 2),
        "y_m": round((float(center[1]) / height) * PITCH_WIDTH_M, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Bonus analytics — pure computation, no API or I/O logic
# ──────────────────────────────────────────────────────────────────────────────

PITCH_ZONE_NAMES: list[str] = ["defensive", "midfield", "attacking"]


def compute_possession_zones(
    tracking_data: list[dict[str, Any]],
    n_zones: int = 3,
) -> dict[str, dict[str, float]]:
    """
    Split the pitch into *n_zones* equal horizontal bands and compute what
    percentage of possession frames each team spent in each zone.

    Orientation assumption: Team A attacks left→right (increasing x), so:
      zone 0  ("defensive")  = low x  → Team A's own half
      zone n-1 ("attacking") = high x → Team A's attacking third

    Args:
        tracking_data: flat tracking list in the standard format.
        n_zones: number of horizontal zones (default 3 = thirds).

    Returns:
        {"A": {"defensive": float, "midfield": float, "attacking": float},
         "B": {...}}
        Values are percentages that sum to 100.0 for each team (or 0.0 if
        the team had no possession frames).
    """
    _, timeline = compute_possession(tracking_data)
    if not timeline:
        return {"A": {}, "B": {}}

    x_vals = [e["center"][0] for e in tracking_data if "center" in e]
    if not x_vals:
        return {"A": {}, "B": {}}

    x_min = min(x_vals)
    x_max = max(x_vals)
    x_range = max(x_max - x_min, 1.0)
    zone_width = x_range / n_zones

    zone_names = (
        PITCH_ZONE_NAMES[:n_zones]
        if n_zones <= len(PITCH_ZONE_NAMES)
        else [f"zone_{i}" for i in range(n_zones)]
    )

    counts: dict[str, Counter] = {"A": Counter(), "B": Counter()}

    for point in timeline:
        team = point.get("team")
        if team not in counts:
            continue
        bx = float(point["ball_center"][0])
        zone_idx = min(int((bx - x_min) / zone_width), n_zones - 1)
        counts[team][zone_names[zone_idx]] += 1

    result: dict[str, dict[str, float]] = {}
    for team in ("A", "B"):
        total = sum(counts[team].values())
        if total == 0:
            result[team] = {z: 0.0 for z in zone_names}
        else:
            result[team] = {
                z: round(counts[team].get(z, 0) / total * 100.0, 1)
                for z in zone_names
            }
    return result


def compute_heatmap_matrix(
    tracking_data: list[dict[str, Any]],
    team: str | None = None,
    rows: int = 10,
    cols: int = 16,
) -> list[list[float]]:
    """
    Build a normalized density heatmap from player positions.

    Pixel coordinates are mapped to a *rows × cols* grid. Each cell accumulates
    one count per tracking entry that falls in it.  The matrix is then
    normalized so the highest-density cell equals 1.0.

    Args:
        tracking_data: flat tracking list.
        team:  filter to "A" or "B"; pass None to include all players.
        rows:  grid rows  (pitch height divisions, default 10).
        cols:  grid columns (pitch width divisions, default 16).

    Returns:
        rows × cols nested list of floats in [0.0, 1.0], row-major order.
        Returns all-zeros if there are no matching player entries.
    """
    empty = [[0.0] * cols for _ in range(rows)]

    players = [
        e for e in tracking_data
        if e.get("class") == "player"
        and (team is None or e.get("team") == team)
    ]
    if not players:
        return empty

    centers = np.array([e["center"] for e in players], dtype=float)
    x_vals = centers[:, 0]
    y_vals = centers[:, 1]

    x_min, x_max = float(x_vals.min()), float(x_vals.max())
    y_min, y_max = float(y_vals.min()), float(y_vals.max())
    x_range = max(x_max - x_min, 1.0)
    y_range = max(y_max - y_min, 1.0)

    matrix = np.zeros((rows, cols), dtype=float)

    col_indices = np.clip(
        ((x_vals - x_min) / x_range * cols).astype(int), 0, cols - 1
    )
    row_indices = np.clip(
        ((y_vals - y_min) / y_range * rows).astype(int), 0, rows - 1
    )

    for r_idx, c_idx in zip(row_indices, col_indices):
        matrix[r_idx, c_idx] += 1.0

    max_val = float(matrix.max())
    if max_val > 0:
        matrix = matrix / max_val

    return [[round(float(v), 4) for v in row] for row in matrix]
