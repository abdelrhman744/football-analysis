"""
team_classifier.py
==================
Self-contained two-stage KMeans jersey-colour classifier.

Stage 1 — Per-crop KMeans(2) isolates the player's jersey colour from the
           background using a corner-pixel majority-vote heuristic.

Stage 2 — Global KMeans(2) clusters all sampled jersey colours into two
           team centroids.  Every subsequent per-player prediction is cached
           by tracker_id so each player is classified at most once.

Adapted from Mohamed20384/Football-Analysis-TactAI/team_assigner/team_assigner.py
and refactored for production use inside the MatchVision pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from sklearn.cluster import KMeans

log = logging.getLogger(__name__)

# Minimum frame appearances before a track_id is included in training.
# Filters out spurious one- or two-frame detections / crowd members.
MIN_TRACK_APPEARANCES: int = 5

# Number of frames sampled at the start of the video for training.
TRAINING_FRAMES: int = 15

# n_init for the *global* (team-level) KMeans — more restarts → more stable.
GLOBAL_N_INIT: int = 20


class TeamClassifier:
    """
    Two-stage KMeans team classifier.

    Usage
    -----
    clf = TeamClassifier()
    clf.train(frames, player_tracks)          # once per video
    team_id = clf.predict(frame, bbox, tid)   # per player per frame
    hex_col  = clf.team_hex_color(team_id)
    conf     = clf.confidence(color_bgr, team_id)
    """

    def __init__(self) -> None:
        # team_id (1 or 2) → KMeans cluster centroid in BGR
        self.team_colors: dict[int, np.ndarray] = {}
        # tracker_id → team_id cache (each player classified only once)
        self._cache: dict[int, int] = {}
        self._kmeans: Optional[KMeans] = None
        # Used to normalise confidence scores
        self._max_intra_dist: float = 1.0
        self._trained: bool = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        frames: list[np.ndarray],
        player_tracks: list[dict[int, dict]],
    ) -> None:
        """
        Fit the global team-colour KMeans on jersey colours sampled from
        the first TRAINING_FRAMES frames.

        Parameters
        ----------
        frames        : list of BGR numpy arrays (must be at least 1 frame)
        player_tracks : list[dict]  indexed by frame_idx.
                        Each dict maps tracker_id → {"bbox": [x1,y1,x2,y2]}
        """
        all_colors: list[np.ndarray] = []
        n_frames = min(TRAINING_FRAMES, len(frames), len(player_tracks))

        for fi in range(n_frames):
            frame = frames[fi]
            for _tid, info in player_tracks[fi].items():
                color = self._jersey_color(frame, info["bbox"])
                if color is not None and color.sum() > 0:
                    all_colors.append(np.array(color, dtype=np.float64))

        if len(all_colors) < 2:
            raise ValueError(
                f"[TeamClassifier] Too few jersey-colour samples ({len(all_colors)}) "
                "to train — check that player detections exist in the first frames."
            )

        km = KMeans(
            n_clusters=2,
            init="k-means++",
            n_init=GLOBAL_N_INIT,
            random_state=0,
        )
        km.fit(all_colors)

        self._kmeans = km
        self.team_colors[1] = km.cluster_centers_[0].copy()
        self.team_colors[2] = km.cluster_centers_[1].copy()

        # Compute max intra-cluster distance for confidence normalisation
        dists: list[float] = []
        for c in all_colors:
            c64 = np.array(c, dtype=np.float64).reshape(1, -1)
            assigned = int(km.predict(c64)[0])
            dists.append(float(np.linalg.norm(c - km.cluster_centers_[assigned])))
        self._max_intra_dist = max(dists, default=1.0) or 1.0

        self._trained = True
        log.info(
            "[TeamClassifier] Trained on %d samples. "
            "Team1 BGR=%s  Team2 BGR=%s",
            len(all_colors),
            np.round(self.team_colors[1]).astype(int).tolist(),
            np.round(self.team_colors[2]).astype(int).tolist(),
        )

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        frame: np.ndarray,
        bbox: list[float],
        tracker_id: int,
    ) -> int:
        """
        Return team_id (1 or 2) for a player.  Cached by tracker_id.

        Raises RuntimeError if train() has not been called first.
        """
        if not self._trained or self._kmeans is None:
            raise RuntimeError("TeamClassifier.train() must be called before predict().")

        if tracker_id in self._cache:
            return self._cache[tracker_id]

        color = self._jersey_color(frame, bbox)
        if color is None or color.sum() == 0:
            # Fallback: assign to whichever team is currently smaller
            team_id = 1 if self._team_size(1) <= self._team_size(2) else 2
        else:
            team_id = int(self._kmeans.predict(
                np.array(color, dtype=np.float64).reshape(1, -1)
            )[0]) + 1

        self._cache[tracker_id] = team_id
        return team_id

    # ── Confidence ────────────────────────────────────────────────────────────

    def confidence(self, color_bgr: np.ndarray, team_id: int) -> float:
        """
        Return a [0.50, 0.99] confidence score based on Euclidean distance
        from the colour to its assigned cluster centroid.
        Closer to centroid → higher confidence.
        """
        if self._kmeans is None or team_id not in self.team_colors:
            return 0.70
        center = self.team_colors[team_id]
        dist = float(np.linalg.norm(np.array(color_bgr, dtype=np.float64) - center))
        conf = max(0.0, 1.0 - dist / max(self._max_intra_dist, 1.0))
        return round(max(0.50, min(0.99, conf)), 3)

    # ── Color helpers ─────────────────────────────────────────────────────────

    def team_hex_color(self, team_id: int) -> str:
        """Return the CSS hex color string for a team (e.g. '#3A6B52')."""
        bgr = self.team_colors.get(team_id, np.array([128, 128, 128]))
        return _bgr_to_hex(bgr)

    def get_jersey_color_for(
        self,
        frame: np.ndarray,
        bbox: list[float],
    ) -> Optional[np.ndarray]:
        """Public wrapper around _jersey_color for external callers."""
        return self._jersey_color(frame, bbox)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _jersey_color(
        self,
        frame: np.ndarray,
        bbox: list[float],
    ) -> Optional[np.ndarray]:
        """
        Extract dominant jersey colour from a player bounding-box crop.

        Steps
        -----
        1. Clamp bbox to frame boundaries.
        2. Take the top half of the crop (torso/jersey, not legs/pitch).
        3. Run KMeans(2) to separate player pixels from background.
        4. Use corner-pixel majority vote to identify the background cluster.
        5. Return the *other* cluster centre = jersey colour (BGR float32).
        """
        try:
            x1, y1, x2, y2 = (int(v) for v in bbox)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                return None

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None

            # Top half = jersey / shirt area
            top_half = crop[: crop.shape[0] // 2, :]
            if top_half.size == 0:
                return None

            pixels = top_half.reshape(-1, 3).astype(np.float64)

            km = KMeans(
                n_clusters=2,
                init="k-means++",
                n_init=1,
                random_state=0,
            )
            km.fit(pixels)

            labels = km.labels_.reshape(top_half.shape[0], top_half.shape[1])
            corners = [
                labels[0, 0],
                labels[0, -1],
                labels[-1, 0],
                labels[-1, -1],
            ]
            bg_cluster = max(set(corners), key=corners.count)
            jersey_cluster = 1 - bg_cluster

            return km.cluster_centers_[jersey_cluster].astype(np.float64)

        except Exception as exc:
            log.debug("[TeamClassifier] _jersey_color error: %s", exc)
            return None

    def _team_size(self, team_id: int) -> int:
        """Number of players already assigned to team_id."""
        return sum(1 for t in self._cache.values() if t == team_id)


# ── Standalone helpers ────────────────────────────────────────────────────────

def _bgr_to_hex(bgr: np.ndarray) -> str:
    """Convert an OpenCV BGR array to a CSS hex string (#RRGGBB)."""
    b, g, r = (int(max(0, min(255, v))) for v in bgr[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


def build_player_tracks_from_detections(
    detections_data: list[dict],
    min_appearances: int = MIN_TRACK_APPEARANCES,
) -> tuple[list[dict[int, dict]], set[int]]:
    """
    Convert the detections JSON produced by video_processor into the
    player_tracks format expected by TeamClassifier.train().

    Parameters
    ----------
    detections_data : raw list loaded from processed_detections.json
    min_appearances : minimum frame count for a track to be included

    Returns
    -------
    player_tracks : list[dict]  length = max(frame index)+1
                   player_tracks[fi][tid] = {"bbox": [x1,y1,x2,y2]}
    stable_ids    : set of tracker_ids that meet the min-appearances threshold
    """
    # Count appearances per tracker_id
    from collections import Counter
    track_counts: Counter = Counter()
    for entry in detections_data:
        for det in entry.get("detections", []):
            tid = det.get("tracker_id")
            if tid is not None:
                track_counts[int(tid)] += 1

    stable_ids = {int(tid) for tid, cnt in track_counts.items() if cnt >= min_appearances}

    # Build per-frame lookup
    n_frames = len(detections_data)
    player_tracks: list[dict[int, dict]] = [{} for _ in range(n_frames)]
    for entry in detections_data:
        fi = int(entry["frame"])
        if fi >= n_frames:
            continue
        for det in entry.get("detections", []):
            tid = det.get("tracker_id")
            if tid is not None and int(tid) in stable_ids:
                player_tracks[fi][int(tid)] = {"bbox": det["bbox"]}

    return player_tracks, stable_ids
