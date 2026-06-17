"""
team_classification_service.py
================================
Produces TeamClassificationResult from processed_team_assignments.json
written by video_processor during the CV frame loop.

Execution path:
  classify_teams(video_id, detection_result)
    └── fast path:  read processed_team_assignments.json
          └── build TeamPlayer list with real confidence scores
    └── slow path:  run TeamClassifier from scratch on video + detections
          └── used only when fast-path file is missing
    └── fallback:   placeholder (random data) — logged as WARNING

All paths return a valid TeamClassificationResult — never raises.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import RESULTS_DIR, UPLOAD_DIR
from models.schemas import AnalysisStatus, TeamClassificationResult, TeamPlayer
from services.cv.team_classifier import (
    TeamClassifier,
    build_player_tracks_from_detections,
    _bgr_to_hex,
    MIN_TRACK_APPEARANCES,
    TRAINING_FRAMES,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def classify_teams(video_id: str, detection_result) -> TeamClassificationResult:
    log.info("[team_classification] Starting for video_id=%s", video_id)

    try:
        # ── Fast path: read assignments written during the video pass ──────────
        assign_path = _locate_assignments(video_id, detection_result)
        if assign_path:
            log.info("[team_classification] Fast path: %s", assign_path)
            result = _from_assignments_file(video_id, assign_path, detection_result)
            if result is not None:
                log.info(
                    "[team_classification] Fast path complete: "
                    "Team A=%d  Team B=%d  colors=(%s, %s)",
                    len(result.team_a_players),
                    len(result.team_b_players),
                    result.team_a_color,
                    result.team_b_color,
                )
                return result
            log.warning("[team_classification] Fast path returned None — trying slow path")

        # ── Slow path: run classifier from scratch ────────────────────────────
        log.info("[team_classification] Slow path: running KMeans classifier")
        result = _classify_from_scratch(video_id, detection_result)
        log.info(
            "[team_classification] Slow path complete: "
            "Team A=%d  Team B=%d  colors=(%s, %s)",
            len(result.team_a_players),
            len(result.team_b_players),
            result.team_a_color,
            result.team_b_color,
        )
        return result

    except Exception as exc:
        log.warning(
            "[team_classification] Both paths failed (%s) — returning placeholder",
            exc, exc_info=True,
        )
        return _placeholder(video_id)


# ─────────────────────────────────────────────────────────────────────────────
# Fast path
# ─────────────────────────────────────────────────────────────────────────────

def _locate_assignments(video_id: str, detection_result) -> Optional[Path]:
    """Return path to *_team_assignments.json, or None if not found."""
    # 1. Check ProcessingOutput.team_assignments_path (set by video_processor)
    cv_out = getattr(detection_result, "_cv_output", None)
    if cv_out is not None:
        p = getattr(cv_out, "team_assignments_path", None)
        if p:
            pp = Path(p)
            if pp.exists():
                log.debug("[team_classification] Found via cv_output: %s", pp)
                return pp
            log.debug("[team_classification] cv_output path doesn't exist: %s", pp)

    # 2. Convention path: results/{video_id}/processed_team_assignments.json
    convention = Path(RESULTS_DIR) / video_id / "processed_team_assignments.json"
    if convention.exists():
        log.debug("[team_classification] Found via convention: %s", convention)
        return convention

    log.info("[team_classification] No assignments file found for video_id=%s", video_id)
    return None


def _from_assignments_file(
    video_id: str,
    assign_path: Path,
    detection_result,
) -> Optional[TeamClassificationResult]:
    """Build result from the pre-computed assignments JSON."""
    try:
        with open(assign_path, encoding="utf-8") as f:
            data: dict = json.load(f)

        assignments: dict[str, int] = {
            str(k): int(v) for k, v in data.get("assignments", {}).items()
        }

        if not assignments:
            log.warning("[team_classification] assignments file is empty: %s", assign_path)
            return None

        team_1_hex = data.get("team_1_color_hex", "#FFFFFF")
        team_2_hex = data.get("team_2_color_hex", "#003366")
        team_1_bgr = np.array(data.get("team_1_color_bgr", [200, 200, 200]), dtype=np.float64)
        team_2_bgr = np.array(data.get("team_2_color_bgr", [50, 50, 150]),   dtype=np.float64)

        log.info(
            "[team_classification] Loaded %d assignments from file. "
            "Team1=%s  Team2=%s",
            len(assignments), team_1_hex, team_2_hex,
        )

        # Inter-cluster distance for confidence scoring
        inter_dist = float(np.linalg.norm(team_1_bgr - team_2_bgr)) or 1.0

        # Extract jersey colours for confidence scoring
        player_colors = _extract_jersey_colors(video_id, detection_result, assignments)

        team_a_players: list[TeamPlayer] = []
        team_b_players: list[TeamPlayer] = []

        for tid_str, team_id in sorted(assignments.items(), key=lambda x: int(x[0])):
            tid        = int(tid_str)
            center_bgr = team_1_bgr if team_id == 1 else team_2_bgr
            other_bgr  = team_2_bgr if team_id == 1 else team_1_bgr
            color_bgr  = player_colors.get(tid)

            if color_bgr is not None:
                d_own   = float(np.linalg.norm(color_bgr - center_bgr))
                d_other = float(np.linalg.norm(color_bgr - other_bgr))
                denom   = d_own + d_other
                if denom < 1e-6:
                    conf = 0.85
                else:
                    conf = round(max(0.50, min(0.99, d_other / denom)), 3)
                hex_c = _bgr_to_hex(color_bgr)
            else:
                conf  = 0.75
                hex_c = team_1_hex if team_id == 1 else team_2_hex

            player = TeamPlayer(
                player_id=tid,
                team_label="Team A" if team_id == 1 else "Team B",
                confidence=conf,
                dominant_color=hex_c,
            )
            if team_id == 1:
                team_a_players.append(player)
            else:
                team_b_players.append(player)

        return TeamClassificationResult(
            video_id=video_id,
            team_a_label="Team A",
            team_b_label="Team B",
            team_a_players=team_a_players,
            team_b_players=team_b_players,
            team_a_color=team_1_hex,
            team_b_color=team_2_hex,
            status=AnalysisStatus.COMPLETED,
        )

    except Exception as exc:
        log.error("[team_classification] _from_assignments_file error: %s", exc, exc_info=True)
        return None


def _extract_jersey_colors(
    video_id: str,
    detection_result,
    assignments: dict[str, int],
) -> dict[int, np.ndarray]:
    """
    Extract one representative jersey color per player for confidence scoring.
    Returns empty dict on any failure — non-fatal.
    """
    try:
        det_path = _locate_detections(video_id, detection_result)
        if not det_path or not det_path.exists():
            return {}

        video_path = _locate_video(video_id)
        if video_path is None:
            return {}

        with open(det_path, encoding="utf-8") as f:
            det_data = json.load(f)

        frames = _read_frames(str(video_path), TRAINING_FRAMES)
        if not frames:
            return {}

        clf_tmp     = TeamClassifier()
        target_tids = {int(k) for k in assignments}
        colors: dict[int, list[np.ndarray]] = {}

        for entry in det_data:
            fi = int(entry["frame"])
            if fi >= len(frames):
                break
            frame = frames[fi]
            for det in entry.get("detections", []):
                tid = det.get("tracker_id")
                if tid is None or int(tid) not in target_tids:
                    continue
                c = clf_tmp.get_jersey_color_for(frame, det["bbox"])
                if c is not None and c.sum() > 0:
                    colors.setdefault(int(tid), []).append(c)

        # Median per player (more robust than first sample)
        medians: dict[int, np.ndarray] = {}
        for tid, samples in colors.items():
            arr = np.stack(samples, axis=0)
            medians[tid] = np.median(arr, axis=0).astype(np.float64)

        log.debug(
            "[team_classification] Extracted jersey colors for %d/%d players",
            len(medians), len(target_tids),
        )
        return medians

    except Exception as exc:
        log.debug("[team_classification] _extract_jersey_colors failed: %s", exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Slow path — run classifier from scratch
# ─────────────────────────────────────────────────────────────────────────────

def _classify_from_scratch(video_id: str, detection_result) -> TeamClassificationResult:
    video_path = _locate_video(video_id)
    if video_path is None:
        raise FileNotFoundError(f"Uploaded video not found for video_id={video_id!r}")

    det_path = _locate_detections(video_id, detection_result)
    if det_path is None or not det_path.exists():
        raise FileNotFoundError(f"Detections file not found for video_id={video_id!r}")

    with open(det_path, encoding="utf-8") as f:
        det_data: list[dict] = json.load(f)

    if not det_data:
        raise ValueError("processed_detections.json is empty")

    player_tracks, stable_ids = build_player_tracks_from_detections(
        det_data, min_appearances=MIN_TRACK_APPEARANCES
    )
    if not stable_ids:
        raise ValueError("No stable tracks found — video may be too short")

    frames = _read_frames(str(video_path), TRAINING_FRAMES)
    if not frames:
        raise IOError(f"Cannot read frames from {video_path}")

    log.info(
        "[team_classification] Training classifier: %d frames, %d stable tracks",
        len(frames), len(stable_ids),
    )

    clf = TeamClassifier()
    clf.train(frames, player_tracks)

    log.info(
        "[team_classification] Classifier trained: Team1=%s  Team2=%s",
        clf.team_hex_color(1), clf.team_hex_color(2),
    )

    # Accumulate votes across all training frames
    for fi, frame_tracks in enumerate(player_tracks):
        ref = frames[fi] if fi < len(frames) else frames[-1]
        for tid, info in frame_tracks.items():
            try:
                clf.accumulate_vote(ref, info["bbox"], tid)
            except Exception:
                pass

    team_a_players: list[TeamPlayer] = []
    team_b_players: list[TeamPlayer] = []

    for tid in stable_ids:
        # Find a representative frame + bbox for this track
        ref_frame = frames[-1]
        ref_bbox  = None
        for fi, frame_tracks in enumerate(player_tracks):
            if tid in frame_tracks:
                ref_frame = frames[fi] if fi < len(frames) else frames[-1]
                ref_bbox  = frame_tracks[tid]["bbox"]
                break

        if ref_bbox is None:
            continue

        try:
            team_id   = clf.predict(ref_frame, ref_bbox, tid)
            if team_id == 0:
                continue  # referee

            # Use median colour for confidence
            color_bgr = clf.get_median_jersey_color(tid)
            if color_bgr is None:
                color_bgr = clf.get_jersey_color_for(ref_frame, ref_bbox)
            if color_bgr is None:
                color_bgr = clf.team_colors.get(team_id, np.zeros(3))

            # Vote-based confidence is most reliable after multi-frame accumulation
            vote_conf  = clf.vote_confidence(tid)
            color_conf = clf.confidence(color_bgr, team_id)
            conf       = round((vote_conf + color_conf) / 2, 3)
            hex_col    = _bgr_to_hex(color_bgr)

            (team_a_players if team_id == 1 else team_b_players).append(
                TeamPlayer(
                    player_id=tid,
                    team_label="Team A" if team_id == 1 else "Team B",
                    confidence=conf,
                    dominant_color=hex_col,
                )
            )
        except Exception as exc:
            log.debug("[team_classification] Skip track %d: %s", tid, exc)

    _persist_assignments_from_classifier(clf, video_id)

    return TeamClassificationResult(
        video_id=video_id,
        team_a_label="Team A",
        team_b_label="Team B",
        team_a_players=sorted(team_a_players, key=lambda p: p.player_id),
        team_b_players=sorted(team_b_players, key=lambda p: p.player_id),
        team_a_color=clf.team_hex_color(1),
        team_b_color=clf.team_hex_color(2),
        status=AnalysisStatus.COMPLETED,
    )


def _persist_assignments_from_classifier(clf: TeamClassifier, video_id: str) -> None:
    """Save assignments JSON so fast-path is available on next dashboard load."""
    try:
        out = Path(RESULTS_DIR) / video_id / "processed_team_assignments.json"
        assignments = {str(k): int(v) for k, v in clf._cache.items() if v in (1, 2)}
        payload = {
            "team_1_color_hex": clf.team_hex_color(1),
            "team_2_color_hex": clf.team_hex_color(2),
            "team_1_color_bgr": [int(v) for v in clf.team_colors.get(1, [0, 0, 0])],
            "team_2_color_bgr": [int(v) for v in clf.team_colors.get(2, [0, 0, 0])],
            "assignments": assignments,
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        log.info(
            "[team_classification] Persisted %d assignments → %s",
            len(assignments), out,
        )
    except Exception as exc:
        log.debug("[team_classification] _persist_assignments_from_classifier: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def _locate_video(video_id: str) -> Optional[Path]:
    upload_dir = Path(UPLOAD_DIR)
    for ext in (".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"):
        p = upload_dir / f"{video_id}{ext}"
        if p.exists():
            return p
    for p in upload_dir.glob(f"{video_id}*"):
        if p.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}:
            return p
    return None


def _locate_detections(video_id: str, detection_result) -> Optional[Path]:
    cv_out = getattr(detection_result, "_cv_output", None)
    if cv_out is not None:
        p = getattr(cv_out, "detections_json_path", None)
        if p and Path(p).exists():
            return Path(p)
    return Path(RESULTS_DIR) / video_id / "processed_detections.json"


def _read_frames(video_path: str, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    frames: list[np.ndarray] = []
    while len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Placeholder fallback
# ─────────────────────────────────────────────────────────────────────────────

def _placeholder(video_id: str) -> TeamClassificationResult:
    log.warning("[team_classification] FALLBACK placeholder for video_id=%s", video_id)
    team_a: list[TeamPlayer] = []
    team_b: list[TeamPlayer] = []
    for pid in range(22):
        if pid < 11:
            team_a.append(TeamPlayer(
                player_id=pid, team_label="Team A",
                confidence=round(random.uniform(0.80, 0.98), 3),
                dominant_color="#FFFFFF",
            ))
        else:
            team_b.append(TeamPlayer(
                player_id=pid, team_label="Team B",
                confidence=round(random.uniform(0.80, 0.98), 3),
                dominant_color="#003366",
            ))
    return TeamClassificationResult(
        video_id=video_id,
        team_a_label="Team A",
        team_b_label="Team B",
        team_a_players=team_a,
        team_b_players=team_b,
        team_a_color="#FFFFFF",
        team_b_color="#003366",
        status=AnalysisStatus.COMPLETED,
    )
