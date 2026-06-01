"""
tracker.py
==========
Multi-object tracker for football players and ball.

The notebook used the `supervision` library's ByteTrack implementation,
which wraps the original ByteTrack algorithm and integrates cleanly with
YOLO detections via supervision's Detections object.

Pipeline (from notebook):
  1. Get raw detections per frame from YOLOv8
  2. Convert to sv.Detections
  3. Pass through sv.ByteTrack → returns Detections with tracker_id assigned
  4. Accumulate per-tracker_id trajectories

TODO: Tune BYTETRACK_* settings in config.py for your video frame rate
      and player density.
"""

from __future__ import annotations
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from services.cv.detector import Detection, FrameDetections

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TrackPoint:
    frame: int
    timestamp: float    # seconds
    cx: float           # centre-x in pixel space
    cy: float           # centre-y in pixel space


@dataclass
class PlayerTrajectory:
    tracker_id: int
    team: Optional[str] = None
    points: List[TrackPoint] = field(default_factory=list)

    # Computed after all frames are processed
    total_distance_px: float = 0.0
    max_speed_px_per_s: float = 0.0
    avg_speed_px_per_s: float = 0.0

    def compute_kinematics(self) -> None:
        """Compute distance and speed from accumulated track points."""
        if len(self.points) < 2:
            return
        dists: List[float] = []
        for a, b in zip(self.points[:-1], self.points[1:]):
            dx = b.cx - a.cx
            dy = b.cy - a.cy
            dt = max(b.timestamp - a.timestamp, 1e-6)
            d  = np.sqrt(dx**2 + dy**2)
            dists.append(d)
            speed = d / dt
            self.max_speed_px_per_s = max(self.max_speed_px_per_s, speed)

        self.total_distance_px = float(np.sum(dists))
        # Average speed over the whole trajectory
        total_time = self.points[-1].timestamp - self.points[0].timestamp
        if total_time > 0:
            self.avg_speed_px_per_s = self.total_distance_px / total_time


@dataclass
class TrackingResult:
    player_trajectories: Dict[int, PlayerTrajectory] = field(default_factory=dict)
    ball_trajectory: List[TrackPoint] = field(default_factory=list)
    total_players_tracked: int = 0


# ---------------------------------------------------------------------------
# Tracker class
# ---------------------------------------------------------------------------

class FootballTracker:
    """
    Wraps supervision's ByteTrack to assign persistent IDs to YOLO detections.

    Usage:
        tracker = FootballTracker()
        for frame_dets in all_frame_detections:
            tracker.update(frame_dets)
        result = tracker.get_result()
    """

    def __init__(self) -> None:
        self._sv_tracker = None
        self._ready = False
        self._player_trajectories: Dict[int, PlayerTrajectory] = defaultdict(
            lambda: PlayerTrajectory(tracker_id=-1)
        )
        self._ball_trajectory: List[TrackPoint] = []
        self._try_init()

    def _try_init(self) -> None:
        try:
            import supervision as sv
            from config import (
                BYTETRACK_TRACK_THRESH,
                BYTETRACK_TRACK_BUFFER,
                BYTETRACK_MATCH_THRESH,
                BYTETRACK_FRAME_RATE,
            )

            # supervision >= 0.18 exposes ByteTrack directly
            self._sv_tracker = sv.ByteTrack(
                track_activation_threshold=BYTETRACK_TRACK_THRESH,
                lost_track_buffer=BYTETRACK_TRACK_BUFFER,
                minimum_matching_threshold=BYTETRACK_MATCH_THRESH,
                frame_rate=BYTETRACK_FRAME_RATE,
            )
            self._ready = True
            log.info("[Tracker] ByteTrack initialised.")

        except ImportError:
            log.warning("[Tracker] supervision not installed — tracking disabled.")
        except Exception as exc:
            log.error("[Tracker] Init failed: %s", exc)

    @property
    def is_ready(self) -> bool:
        return self._ready

    def update(self, frame_dets: FrameDetections) -> FrameDetections:
        """
        Feed one frame's detections through ByteTrack.
        Returns the same FrameDetections object with tracker_id filled in.
        Updates internal trajectory accumulators.

        TODO: The notebook called tracker.update_with_detections(sv_dets) and
              then annotated frames with sv.BoundingBoxAnnotator + sv.LabelAnnotator.
              The annotated frames are what get written to the output video in
              video_processor.py — see annotate_frame() there.
        """
        if not self._ready or self._sv_tracker is None:
            # No tracking — still accumulate ball positions if available
            if frame_dets.ball:
                self._ball_trajectory.append(
                    TrackPoint(
                        frame=frame_dets.frame_index,
                        timestamp=frame_dets.timestamp,
                        cx=frame_dets.ball.bbox.cx,
                        cy=frame_dets.ball.bbox.cy,
                    )
                )
            return frame_dets

        try:
            import supervision as sv

            all_dets = frame_dets.players + frame_dets.referees

            if not all_dets:
                return frame_dets

            xyxy       = np.array([[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in all_dets])
            confidence = np.array([d.confidence for d in all_dets])
            class_ids  = np.array([d.class_id   for d in all_dets])

            sv_dets = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_ids,
            )

            # ── This is the key ByteTrack call from the notebook ───────────
            tracked = self._sv_tracker.update_with_detections(sv_dets)
            # ─────────────────────────────────────────────────────────────

            if tracked.tracker_id is None:
                return frame_dets

            # Map tracker_id back onto our Detection objects
            for i, (x1, y1, x2, y2) in enumerate(tracked.xyxy):
                tid = int(tracked.tracker_id[i])
                cid = int(tracked.class_id[i]) if tracked.class_id is not None else 0

                # Find matching detection by bbox overlap
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                for det in frame_dets.players + frame_dets.referees:
                    if abs(det.bbox.cx - cx) < 5 and abs(det.bbox.cy - cy) < 5:
                        det.tracker_id = tid
                        break

                # Accumulate player trajectory
                traj = self._player_trajectories[tid]
                traj.tracker_id = tid
                traj.points.append(
                    TrackPoint(
                        frame=frame_dets.frame_index,
                        timestamp=frame_dets.timestamp,
                        cx=float(cx),
                        cy=float(cy),
                    )
                )

        except Exception as exc:
            log.error("[Tracker] update() error on frame %d: %s", frame_dets.frame_index, exc)

        # Ball tracking (simple, no ByteTrack needed for single object)
        if frame_dets.ball:
            self._ball_trajectory.append(
                TrackPoint(
                    frame=frame_dets.frame_index,
                    timestamp=frame_dets.timestamp,
                    cx=frame_dets.ball.bbox.cx,
                    cy=frame_dets.ball.bbox.cy,
                )
            )

        return frame_dets

    def get_result(self) -> TrackingResult:
        """Finalise trajectories and return summary result."""
        for traj in self._player_trajectories.values():
            traj.compute_kinematics()

        return TrackingResult(
            player_trajectories=dict(self._player_trajectories),
            ball_trajectory=self._ball_trajectory,
            total_players_tracked=len(self._player_trajectories),
        )

    def reset(self) -> None:
        """Clear state between videos."""
        if self._sv_tracker:
            try:
                self._sv_tracker.reset()
            except Exception:
                pass
        self._player_trajectories = defaultdict(
            lambda: PlayerTrajectory(tracker_id=-1)
        )
        self._ball_trajectory = []
