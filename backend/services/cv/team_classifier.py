"""
team_classifier.py
==================
Improved two-stage KMeans jersey-colour classifier.

Improvements over the original:
  * HSV-based grass-green masking before colour extraction (removes pitch pixels)
  * Upper-body crop refined to 20-60% of bbox height (skips head & legs)
  * Median colour per player rather than KMeans cluster centre (more robust)
  * Multi-frame voting: each tracker_id accumulates colour samples across frames
    and a majority vote determines the final team assignment
  * Temporal smoothing via vote accumulation dict (_votes)
  * Referee filtering: players whose median colour is close to black/grey are
    excluded from training and tagged as referee
  * Inter-cluster confidence: score based on ratio of distance to own centroid
    vs distance to opposite centroid (proper discriminant confidence)
  * Team assignment persistence via _cache — still one prediction per player
    after votes are finalised
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np
from sklearn.cluster import KMeans

log = logging.getLogger(__name__)

# ── Tuneable constants ────────────────────────────────────────────────────────

MIN_TRACK_APPEARANCES: int = 5   # frames a track must appear before training
TRAINING_FRAMES: int = 30        # frames sampled for training (was 15)
GLOBAL_N_INIT: int = 25          # restarts for the global KMeans

# HSV range for grass-green masking (removes pitch pixels from crop)
_GRASS_LOWER = np.array([35,  40,  40], dtype=np.uint8)
_GRASS_UPPER = np.array([90, 255, 255], dtype=np.uint8)

# Referee detection: jerseys whose saturation is very low (grey/black/white)
_REFEREE_SAT_THRESHOLD: int = 40   # HSV S channel
_REFEREE_LIGHTNESS_THRESHOLD: int = 200  # very bright → white kit → still usable, not filtered

# Minimum number of valid colour samples to trust a player's colour
_MIN_COLOUR_SAMPLES: int = 3

# Votes needed per player before we lock in their team assignment
_MIN_VOTES: int = 3


class TeamClassifier:
    """
    Two-stage KMeans team classifier with multi-frame voting.

    Usage
    -----
    clf = TeamClassifier()
    clf.train(frames, player_tracks)          # once per video
    clf.accumulate_vote(frame, bbox, tid)     # called every frame in the pipeline
    team_id = clf.predict(frame, bbox, tid)   # returns cached result or votes majority
    hex_col  = clf.team_hex_color(team_id)
    conf     = clf.confidence(color_bgr, team_id)
    """

    def __init__(self) -> None:
        # team_id (1 or 2) → KMeans cluster centroid in BGR
        self.team_colors: dict[int, np.ndarray] = {}
        # tracker_id → team_id cache (locked after MIN_VOTES votes)
        self._cache: dict[int, int] = {}
        # tracker_id → {team_id: vote_count}
        self._votes: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        # tracker_id → list of jersey BGR colours (for multi-frame median)
        self._colour_samples: dict[int, list[np.ndarray]] = defaultdict(list)
        self._kmeans: Optional[KMeans] = None
        self._inter_dist: float = 1.0
        self._trained: bool = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        frames: list[np.ndarray],
        player_tracks: list[dict[int, dict]],
    ) -> None:
        """
        Fit the global team-colour KMeans on jersey colours sampled from the
        first TRAINING_FRAMES frames, excluding referee-like colours.
        """
        all_colors: list[np.ndarray] = []
        n_frames = min(TRAINING_FRAMES, len(frames), len(player_tracks))

        for fi in range(n_frames):
            frame = frames[fi]
            for _tid, info in player_tracks[fi].items():
                color = self._extract_jersey_color(frame, info["bbox"])
                if color is None:
                    continue
                if _is_referee_color(color):
                    continue
                all_colors.append(color.astype(np.float64))

        if len(all_colors) < 2:
            raise ValueError(
                f"[TeamClassifier] Too few usable jersey-colour samples ({len(all_colors)}) "
                "after referee filtering — check player detections in training frames."
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
        self._inter_dist = float(
            np.linalg.norm(self.team_colors[1] - self.team_colors[2])
        ) or 1.0

        self._trained = True
        log.info(
            "[TeamClassifier] Trained on %d samples. "
            "Team1 BGR=%s  Team2 BGR=%s  inter_dist=%.1f",
            len(all_colors),
            np.round(self.team_colors[1]).astype(int).tolist(),
            np.round(self.team_colors[2]).astype(int).tolist(),
            self._inter_dist,
        )

    # ── Vote accumulation (call every frame in the pipeline) ─────────────────

    def accumulate_vote(
        self,
        frame: np.ndarray,
        bbox: list[float],
        tracker_id: int,
    ) -> None:
        """
        Extract this player's jersey colour and cast a vote for the matching
        team.  Skips referees and unclassifiable crops silently.
        Does nothing if the tracker_id is already locked in _cache.
        """
        if not self._trained or self._kmeans is None:
            return
        if tracker_id in self._cache:
            return

        color = self._extract_jersey_color(frame, bbox)
        if color is None:
            return
        if _is_referee_color(color):
            self._cache[tracker_id] = 0  # 0 = referee sentinel
            return

        self._colour_samples[tracker_id].append(color)

        team_id = int(self._kmeans.predict(
            color.astype(np.float64).reshape(1, -1)
        )[0]) + 1
        self._votes[tracker_id][team_id] += 1

        # Lock assignment once we have enough votes
        total_votes = sum(self._votes[tracker_id].values())
        if total_votes >= _MIN_VOTES:
            best = max(self._votes[tracker_id], key=self._votes[tracker_id].get)
            self._cache[tracker_id] = best

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        frame: np.ndarray,
        bbox: list[float],
        tracker_id: int,
    ) -> int:
        """
        Return team_id (1 or 2) for a player.

        Priority:
          1. Locked cache value (set after MIN_VOTES votes)
          2. Current vote majority if any votes exist
          3. Single-frame KMeans prediction as fallback
          4. Balance-based fallback if jersey unreadable

        Raises RuntimeError if train() has not been called first.
        """
        if not self._trained or self._kmeans is None:
            raise RuntimeError("TeamClassifier.train() must be called before predict().")

        # Referee sentinel
        if self._cache.get(tracker_id) == 0:
            return 1  # assign referee to team 1 silently; caller can filter

        if tracker_id in self._cache:
            return self._cache[tracker_id]

        # Use accumulated votes if available
        if self._votes[tracker_id]:
            team_id = max(self._votes[tracker_id], key=self._votes[tracker_id].get)
            self._cache[tracker_id] = team_id
            return team_id

        # Single-frame fallback
        color = self._extract_jersey_color(frame, bbox)
        if color is None or _is_referee_color(color):
            team_id = 1 if self._team_size(1) <= self._team_size(2) else 2
        else:
            team_id = int(self._kmeans.predict(
                color.astype(np.float64).reshape(1, -1)
            )[0]) + 1
            self.accumulate_vote(frame, bbox, tracker_id)

        self._cache[tracker_id] = team_id
        return team_id

    # ── Confidence ────────────────────────────────────────────────────────────

    def confidence(self, color_bgr: np.ndarray, team_id: int) -> float:
        """
        Return a [0.50, 0.99] confidence score using inter-cluster ratio.

        Score = distance_to_opposite / (distance_to_own + distance_to_opposite)
        This properly reflects how separable this sample is between the two teams.
        """
        if self._kmeans is None or team_id not in self.team_colors:
            return 0.70

        other_id = 2 if team_id == 1 else 1
        c = np.array(color_bgr, dtype=np.float64)
        d_own   = float(np.linalg.norm(c - self.team_colors[team_id]))
        d_other = float(np.linalg.norm(c - self.team_colors.get(other_id, c)))

        denom = d_own + d_other
        if denom < 1e-6:
            return 0.85  # identical to both centroids (degenerate case)

        # High ratio = far from own centroid → lower confidence; invert it
        conf = d_other / denom
        return round(max(0.50, min(0.99, conf)), 3)

    def vote_confidence(self, tracker_id: int) -> float:
        """
        Return vote-based confidence for a locked tracker_id.
        = (majority_votes) / (total_votes).
        Falls back to 0.75 if no votes recorded.
        """
        votes = self._votes.get(tracker_id)
        if not votes:
            return 0.75
        total = sum(votes.values())
        if total == 0:
            return 0.75
        best_count = max(votes.values())
        raw = best_count / total
        return round(max(0.50, min(0.99, 0.5 + raw * 0.49)), 3)

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
        """Public wrapper around _extract_jersey_color for external callers."""
        return self._extract_jersey_color(frame, bbox)

    def get_median_jersey_color(self, tracker_id: int) -> Optional[np.ndarray]:
        """Return the median of all collected colour samples for a tracker_id."""
        samples = self._colour_samples.get(tracker_id)
        if not samples:
            return None
        arr = np.stack(samples, axis=0)
        return np.median(arr, axis=0).astype(np.float64)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_jersey_color(
        self,
        frame: np.ndarray,
        bbox: list[float],
    ) -> Optional[np.ndarray]:
        """
        Extract dominant jersey colour from a player bounding-box crop.

        Steps
        -----
        1. Clamp bbox to frame boundaries.
        2. Take the 20–60% vertical band of the crop (torso only, not head or legs).
        3. Convert to HSV; mask out grass-green pixels.
        4. If too few non-grass pixels remain, use the unmasked torso.
        5. Compute the median BGR colour of the remaining pixels.
        6. Return None if the crop is too small or entirely masked.
        """
        try:
            x1, y1, x2, y2 = (int(v) for v in bbox)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 - x1 < 8 or y2 - y1 < 16:
                return None

            crop = frame[y1:y2, x1:x2]
            ch   = crop.shape[0]

            # Torso band: 20–60% of bbox height
            top    = max(0, int(ch * 0.20))
            bottom = min(ch, int(ch * 0.60))
            torso  = crop[top:bottom, :]

            if torso.size == 0:
                return None

            # Mask grass pixels in HSV space
            torso_hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
            grass_mask = cv2.inRange(torso_hsv, _GRASS_LOWER, _GRASS_UPPER)
            non_grass  = cv2.bitwise_not(grass_mask)

            pixels_bgr = torso.reshape(-1, 3)
            mask_flat  = non_grass.reshape(-1)

            usable = pixels_bgr[mask_flat > 0]
            if len(usable) < _MIN_COLOUR_SAMPLES:
                # Fallback: use all torso pixels without grass mask
                usable = pixels_bgr

            if len(usable) == 0:
                return None

            # Median colour is more robust to outlier pixels than mean
            median_bgr = np.median(usable, axis=0).astype(np.float64)
            return median_bgr

        except Exception as exc:
            log.debug("[TeamClassifier] _extract_jersey_color error: %s", exc)
            return None

    def _team_size(self, team_id: int) -> int:
        """Number of players already assigned to team_id (excluding referees)."""
        return sum(1 for t in self._cache.values() if t == team_id)


# ── Standalone helpers ────────────────────────────────────────────────────────

def _is_referee_color(bgr: np.ndarray) -> bool:
    """
    Return True if the colour looks like a referee kit (very low saturation).
    Referees typically wear black, dark grey, or fluorescent yellow.
    We only filter low-saturation (achromatic) colours here.
    """
    bgr_u8 = np.array([[bgr[:3].astype(np.uint8)]], dtype=np.uint8)
    hsv    = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV)[0, 0]
    s      = int(hsv[1])
    return s < _REFEREE_SAT_THRESHOLD


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
    """
    from collections import Counter
    track_counts: Counter = Counter()
    for entry in detections_data:
        for det in entry.get("detections", []):
            tid = det.get("tracker_id")
            if tid is not None:
                track_counts[int(tid)] += 1

    stable_ids = {int(tid) for tid, cnt in track_counts.items() if cnt >= min_appearances}

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
