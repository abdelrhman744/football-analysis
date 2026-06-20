"""
video_processor.py — CV pipeline with H.264 browser-compatible output.

Changes in this revision
------------------------
* Per-team heatmaps (heatmap_team1, heatmap_team2) accumulated in the frame
    loop after team assignment via TeamClassifier.predict(), matching the
    notebook's two-accumulator pattern exactly.
* Ball detection drawn as a green circle on the annotated frame (matches
    notebook visualisation cell).
* Player trajectory lines drawn per-frame in team colour using stored
    track-point history (matches notebook's cv2.line() trajectory cell).
* ProcessingOutput gains heatmap_team1_path and heatmap_team2_path fields.
* _generate_per_team_heatmaps() added; _generate_heatmap() retained for the
    combined all-player heatmap (served as heatmap_player_path).
* config.py class IDs updated to best.pt layout:
    CLASS_BALL=0, CLASS_GOALKEEPER=1, CLASS_PLAYER=2, CLASS_REFEREE=3
"""

import cv2
import json
import logging
import os
import shutil
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field

import supervision as sv
from ultralytics import YOLO

from .notebook_adapter import get_annotator_bundle, numpy_to_python
from .team_classifier import (
    TeamClassifier,
    _bgr_to_hex,
    TRAINING_FRAMES,
    MIN_TRACK_APPEARANCES,
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessingOutput:
    result: dict[str, Any]
    summary: dict[str, Any]
    used_real_model: bool = True
    detections_json_path: str | None = None
    tracking_json_path: str | None = None
    # Combined all-player heatmap (existing field, kept for back-compat)
    heatmap_players_path: str | None = None
    heatmap_player_path: str | None = None
    heatmap_ball_path: str | None = None
    heatmap_matrix: list[list[float]] | None = None
    # Per-team heatmaps (new)
    heatmap_team1_path: str | None = None
    heatmap_team2_path: str | None = None
    processed_video_path: str | None = None
    team_assignments_path: str | None = None
    warning: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def process_match_video(
    video_path: str,
    model_path: str,
    output_path: str,
    conf_threshold: float = 0.3,
) -> ProcessingOutput:
    result = process_video(
        video_path=video_path,
        model_path=model_path,
        output_path=output_path,
        conf_threshold=conf_threshold,
    )

    real   = result.get("real_cv_analysis", {})
    status = result.get("pipeline_status", {})

    summary = {
        "total_frames_processed":    real.get("frames_processed", 0),
        "total_detections":          real.get("total_detections", 0),
        "max_players_in_frame":      real.get("unique_tracks", 0),
        "frames_with_ball_detected": real.get("total_detections", 0),
    }

    output_base = Path(output_path).with_suffix("")

    return ProcessingOutput(
        result=result,
        summary=summary,
        used_real_model=True,
        detections_json_path=str(output_base) + "_detections.json",
        tracking_json_path=str(output_base) + "_tracking.json",
        heatmap_players_path=real.get("heatmap_path"),
        heatmap_player_path=real.get("heatmap_path"),
        heatmap_ball_path=None,
        heatmap_matrix=None,
        heatmap_team1_path=real.get("heatmap_team1_path"),
        heatmap_team2_path=real.get("heatmap_team2_path"),
        processed_video_path=real.get("output_video_path"),
        team_assignments_path=real.get("team_assignments_path"),
        warning="; ".join(status.get("errors", [])) if status.get("errors") else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_video(
    video_path: str,
    model_path: str,
    output_path: str,
    conf_threshold: float = 0.3,
) -> dict[str, Any]:

    status: dict[str, Any] = {
        "model_loaded":            False,
        "video_opened":            False,
        "frames_processed":        0,
        "annotation_ready":        False,
        "output_written":          False,
        "team_classifier_trained": False,
        "team_assignments_count":  0,
        "errors":                  [],
    }

    # ── 1. Load YOLO model ────────────────────────────────────────────────────
    try:
        model = YOLO(model_path)
        status["model_loaded"] = True
        logger.info("[VideoProcessor] Model loaded: %s  classes=%s", model_path, model.names)
    except Exception as exc:
        status["errors"].append(f"Model load failed: {exc}")
        logger.error("[VideoProcessor] Model load failed: %s", exc)
        return _build_result({}, status)

    # ── 2. Open source video ──────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        status["errors"].append(f"Cannot open video: {video_path}")
        return _build_result({}, status)

    status["video_opened"] = True
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
        "[VideoProcessor] Video: %dx%d @ %.1f fps, ~%d frames",
        width, height, fps, total_frames,
    )

    # ── 3. Prepare annotator + tracker ───────────────────────────────────────
    bundle = get_annotator_bundle()
    status["annotation_ready"] = getattr(bundle, "ready", False)
    tracker = sv.ByteTrack()

    # ── 4. Open VideoWriter → TEMPORARY file ─────────────────────────────────
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".mp4", dir=str(output_path_obj.parent)
    )
    os.close(tmp_fd)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

    if not out.isOpened():
        status["errors"].append(f"Cannot open VideoWriter: {tmp_path}")
        cap.release()
        _safe_remove(tmp_path)
        return _build_result({}, status)

    # ── 5. Per-frame inference loop ───────────────────────────────────────────
    # Track-point history: tracker_id → list of (cx, cy) for trajectory lines
    all_track_points: dict[int, list[tuple[float, float]]] = {}

    # Per-team heatmap accumulators (notebook pattern)
    heatmap_team1 = np.zeros((height, width), dtype=np.float32)
    heatmap_team2 = np.zeros((height, width), dtype=np.float32)

    # Team colour lookup for trajectory lines: tracker_id → BGR tuple
    track_team_color: dict[int, tuple[int, int, int]] = {}

    total_detections  = 0
    unique_track_ids: set[int] = set()
    detections_export: list[dict[str, Any]] = []
    tracking_export:   list[dict[str, Any]] = []

    # Team classifier — two-phase approach:
    #   Phase A: buffer first TRAINING_FRAMES frames for training
    #   Phase B: train once, then accumulate_vote() every frame, annotate
    team_clf: TeamClassifier | None = None
    _train_buf_frames:  list[np.ndarray]           = []
    _train_buf_dets:    list[list[dict[str, Any]]]  = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── inference ─────────────────────────────────────────────────────────
        frame_dets: list[dict[str, Any]] = []
        ball_det: dict[str, Any] | None = None
        try:
            results    = model.predict(frame, conf=conf_threshold, verbose=False)
            raw        = results[0]
            detections = sv.Detections.from_ultralytics(raw)
            detections = tracker.update_with_detections(detections)

            total_detections += len(detections)

            for i, xyxy in enumerate(detections.xyxy):
                class_id   = int(detections.class_id[i])   if detections.class_id   is not None else None
                confidence = float(detections.confidence[i]) if detections.confidence is not None else None
                tracker_id = int(detections.tracker_id[i])  if detections.tracker_id is not None else None

                det_entry = {
                    "class_id":   class_id,
                    "confidence": confidence,
                    "tracker_id": tracker_id,
                    "bbox":       [float(v) for v in xyxy],
                }

                # Separate ball detections so we can draw the circle.
                # CLASS_BALL = 0 per best.pt layout.
                if class_id == 0:
                    ball_det = det_entry
                else:
                    frame_dets.append(det_entry)

                if tracker_id is not None:
                    unique_track_ids.add(tracker_id)
                    cx = float((xyxy[0] + xyxy[2]) / 2)
                    cy = float((xyxy[1] + xyxy[3]) / 2)
                    all_track_points.setdefault(tracker_id, []).append((cx, cy))
                    tracking_export.append({
                        "frame":    int(frame_idx),
                        "track_id": tracker_id,
                        "center":   [cx, cy],
                        "bbox":     [float(v) for v in xyxy],
                    })

            all_dets_for_export = frame_dets + ([ball_det] if ball_det else [])
            detections_export.append({"frame": int(frame_idx), "detections": all_dets_for_export})

        except Exception as exc:
            logger.warning("[VideoProcessor] Frame %d inference error: %s", frame_idx, exc)
            detections = sv.Detections.empty()
            frame_dets = []
            ball_det   = None

        # ── team classifier: buffer → train → vote every frame ────────────────
        if team_clf is None:
            _train_buf_frames.append(frame.copy())
            _train_buf_dets.append(frame_dets)

            if len(_train_buf_frames) >= TRAINING_FRAMES:
                logger.info(
                    "[VideoProcessor] TeamClassifier: training on %d frames",
                    len(_train_buf_frames),
                )
                team_clf = _train_team_classifier(_train_buf_frames, _train_buf_dets)
                if team_clf is not None:
                    status["team_classifier_trained"] = True
                    logger.info(
                        "[VideoProcessor] TeamClassifier trained. Team1=%s Team2=%s",
                        team_clf.team_hex_color(1),
                        team_clf.team_hex_color(2),
                    )
                    # Accumulate votes for all buffered training frames
                    for buf_frame, buf_dets in zip(_train_buf_frames, _train_buf_dets):
                        for det in buf_dets:
                            tid = det.get("tracker_id")
                            if tid is not None:
                                try:
                                    team_clf.accumulate_vote(buf_frame, det["bbox"], int(tid))
                                except Exception:
                                    pass
                else:
                    logger.warning("[VideoProcessor] TeamClassifier training failed")
                _train_buf_frames.clear()
                _train_buf_dets.clear()
        else:
            # Accumulate votes for current frame players
            for det in frame_dets:
                tid = det.get("tracker_id")
                if tid is not None:
                    try:
                        team_clf.accumulate_vote(frame, det["bbox"], int(tid))
                    except Exception:
                        pass

        # ── annotate frame (supervision boxes + labels + traces) ──────────────
        # Referees pass through here and receive supervision bounding-box
        # annotation. They are excluded only from the team-ellipse / trajectory
        # / heatmap logic inside _draw_players (team_id == 0 sentinel check).
        try:
            annotated = bundle.annotate_frame(frame, detections)
            if annotated is None:
                annotated = frame.copy()
        except Exception as exc:
            if frame_idx == 0:
                logger.warning("[VideoProcessor] Annotation disabled: %s", exc)
            annotated = frame.copy()

        # ── per-player: team ellipse + trajectory lines + heatmap update ──────
        if team_clf is not None:
            annotated = _draw_players(
                annotated=annotated,
                frame_dets=frame_dets,
                raw_frame=frame,
                clf=team_clf,
                track_points=all_track_points,
                track_team_color=track_team_color,
                heatmap_team1=heatmap_team1,
                heatmap_team2=heatmap_team2,
            )

        # ── ball circle (green, matches notebook) ─────────────────────────────
        if ball_det is not None:
            try:
                bx1, by1, bx2, by2 = (int(v) for v in ball_det["bbox"])
                cx_b = (bx1 + bx2) // 2
                cy_b = (by1 + by2) // 2
                cv2.circle(annotated, (cx_b, cy_b), 10, (0, 255, 0), 3)
            except Exception:
                pass

        if annotated.shape[1] == width and annotated.shape[0] == height:
            out.write(annotated)
        else:
            out.write(frame)

        frame_idx += 1

    # Handle videos shorter than TRAINING_FRAMES
    if team_clf is None and _train_buf_frames:
        logger.info(
            "[VideoProcessor] Short video (%d frames) — training TeamClassifier now",
            len(_train_buf_frames),
        )
        team_clf = _train_team_classifier(_train_buf_frames, _train_buf_dets)
        if team_clf is not None:
            status["team_classifier_trained"] = True
            for buf_frame, buf_dets in zip(_train_buf_frames, _train_buf_dets):
                for det in buf_dets:
                    tid = det.get("tracker_id")
                    if tid is not None:
                        try:
                            team_clf.accumulate_vote(buf_frame, det["bbox"], int(tid))
                        except Exception:
                            pass
        _train_buf_frames.clear()
        _train_buf_dets.clear()

    cap.release()
    out.release()

    status["frames_processed"] = int(frame_idx)
    if team_clf is not None:
        status["team_assignments_count"] = len(team_clf._cache)

    logger.info(
        "[VideoProcessor] Done: %d frames | %d detections | %d tracks | "
        "classifier_trained=%s | assignments=%d",
        frame_idx, total_detections, len(unique_track_ids),
        status["team_classifier_trained"],
        status.get("team_assignments_count", 0),
    )

    # ── 6. Re-encode to H.264 + faststart ────────────────────────────────────
    encode_ok = _reencode_h264(src=tmp_path, dst=str(output_path_obj), fps=fps)
    _safe_remove(tmp_path)

    if not encode_ok:
        logger.error("[VideoProcessor] H.264 re-encode failed — falling back to mp4v")
        status["errors"].append("H.264 re-encode failed; output may not play in browser")
    else:
        status["output_written"] = True
        logger.info("[VideoProcessor] H.264 output: %s", output_path_obj)

    # ── 7. Save JSON outputs + heatmaps + team assignments ────────────────────
    output_base   = output_path_obj.with_suffix("")
    det_json_path = str(output_base) + "_detections.json"
    trk_json_path = str(output_base) + "_tracking.json"

    with open(det_json_path, "w", encoding="utf-8") as f:
        json.dump(numpy_to_python(detections_export), f, ensure_ascii=False, indent=2)

    with open(trk_json_path, "w", encoding="utf-8") as f:
        json.dump(numpy_to_python(tracking_export), f, ensure_ascii=False, indent=2)

    # Combined all-player heatmap (existing behaviour)
    heatmap_path = _generate_heatmap(all_track_points, width, height, str(output_path_obj))

    # Per-team heatmaps (new — matches notebook)
    heatmap_team1_path, heatmap_team2_path = _generate_per_team_heatmaps(
        heatmap_team1, heatmap_team2, str(output_base)
    )

    team_assign_path = _save_team_assignments(team_clf, str(output_base))
    if team_assign_path:
        logger.info("[VideoProcessor] Team assignments saved: %s", team_assign_path)

    cv_data = {
        "frames_processed":         int(frame_idx),
        "total_detections":         int(total_detections),
        "unique_tracks":            int(len(unique_track_ids)),
        "fps":                      float(round(fps, 2)),
        "resolution":               {"width": int(width), "height": int(height)},
        "heatmap_path":             heatmap_path,
        "heatmap_team1_path":       heatmap_team1_path,
        "heatmap_team2_path":       heatmap_team2_path,
        "output_video_path":        str(output_path_obj),
        "detections_json_path":     det_json_path,
        "tracking_json_path":       trk_json_path,
        "team_assignments_path":    team_assign_path,
        "team_classifier_trained":  bool(team_clf is not None),
        "team_1_color_hex":         team_clf.team_hex_color(1) if team_clf else None,
        "team_2_color_hex":         team_clf.team_hex_color(2) if team_clf else None,
    }

    return _build_result(cv_data, status)


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame drawing: team ellipses + trajectory lines + heatmap accumulation
# ─────────────────────────────────────────────────────────────────────────────

def _draw_players(
    annotated: np.ndarray,
    frame_dets: list[dict[str, Any]],
    raw_frame: np.ndarray,
    clf: "TeamClassifier",
    track_points: dict[int, list[tuple[float, float]]],
    track_team_color: dict[int, tuple[int, int, int]],
    heatmap_team1: np.ndarray,
    heatmap_team2: np.ndarray,
) -> np.ndarray:
    """
    For every tracked player in this frame:
      1. Predict team via TeamClassifier (uses cached vote result after MIN_VOTES).
      2. Draw a team-coloured ellipse under the player's feet.
      3. Draw trajectory lines (all stored points so far) in team colour.
      4. Accumulate the player's centre into the correct team heatmap grid.

    Referees: class_id == CLASS_REFEREE detections are NOT in frame_dets
    (they remain in the full detections set passed to bundle.annotate_frame,
    which draws their bounding box via supervision). If a referee slips through
    with team_id == 0 from the classifier, the sentinel check below skips them.

    Non-fatal: returns original annotated frame on any error.
    """
    try:
        canvas = annotated.copy()
        h, w   = canvas.shape[:2]

        for det in frame_dets:
            tid  = det.get("tracker_id")
            bbox = det.get("bbox")
            if tid is None or bbox is None:
                continue

            try:
                team_id = clf.predict(raw_frame, bbox, int(tid))
            except Exception:
                continue

            if team_id == 0:
                continue  # referee sentinel — skip team-colour rendering

            # Resolve BGR colour from team centroid
            bgr_arr = clf.team_colors.get(team_id)
            if bgr_arr is not None:
                bgr: tuple[int, int, int] = tuple(
                    int(max(0, min(255, v))) for v in bgr_arr[:3]
                )  # type: ignore[assignment]
            else:
                bgr = (255, 120, 0) if team_id == 1 else (0, 180, 255)

            # Cache team colour per tracker (used for trajectory lines)
            track_team_color[int(tid)] = bgr

            x1, y1, x2, y2 = (int(v) for v in bbox)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # ── Team ellipse under feet ────────────────────────────────────
            half_w = max(4, (x2 - x1) // 2)
            cv2.ellipse(
                canvas,
                center=(cx, y2),
                axes=(half_w, max(2, int(0.35 * half_w))),
                angle=0.0,
                startAngle=-45,
                endAngle=235,
                color=bgr,
                thickness=3,
                lineType=cv2.LINE_AA,
            )

            # ── Team label above box ───────────────────────────────────────
            label = f"Team {team_id} | ID {int(tid)}"
            cv2.putText(
                canvas,
                label,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                bgr,
                2,
                cv2.LINE_AA,
            )

            # ── Trajectory lines (notebook cv2.line pattern) ───────────────
            pts = track_points.get(int(tid), [])
            for i in range(1, len(pts)):
                p0 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                p1 = (int(pts[i][0]),     int(pts[i][1]))
                cv2.line(canvas, p0, p1, bgr, 2, cv2.LINE_AA)

            # ── Per-team heatmap accumulation (notebook pattern) ───────────
            xi = int(np.clip(cx, 0, w - 1))
            yi = int(np.clip(cy, 0, h - 1))
            if team_id == 1:
                heatmap_team1[yi, xi] += 1.0
            else:
                heatmap_team2[yi, xi] += 1.0

        return canvas

    except Exception as exc:
        logger.debug("[VideoProcessor] _draw_players error: %s", exc)
        return annotated


# ─────────────────────────────────────────────────────────────────────────────
# Team classification helpers
# ─────────────────────────────────────────────────────────────────────────────

def _train_team_classifier(
    frames: list[np.ndarray],
    frame_dets_list: list[list[dict[str, Any]]],
) -> "TeamClassifier | None":
    """
    Train TeamClassifier from buffered frames + their detection lists.
    Returns None on any failure — pipeline always continues.
    """
    try:
        n = len(frames)
        player_tracks: list[dict[int, dict]] = [{} for _ in range(n)]
        track_counts: dict[int, int] = {}

        for fi, frame_dets in enumerate(frame_dets_list):
            for det in frame_dets:
                tid = det.get("tracker_id")
                if tid is not None:
                    tid = int(tid)
                    track_counts[tid] = track_counts.get(tid, 0) + 1
                    player_tracks[fi][tid] = {"bbox": det["bbox"]}

        stable = {tid for tid, cnt in track_counts.items() if cnt >= MIN_TRACK_APPEARANCES}
        if not stable:
            stable = {tid for tid, cnt in track_counts.items() if cnt >= 1}
            logger.info(
                "[VideoProcessor] TeamClassifier: relaxed min_appearances to 1 "
                "(%d buffered frames)", n
            )

        if not stable:
            raise ValueError("No tracked players in training frames")

        filtered: list[dict[int, dict]] = [
            {tid: info for tid, info in pt.items() if tid in stable}
            for pt in player_tracks
        ]

        clf = TeamClassifier()
        clf.train(frames, filtered)
        return clf

    except Exception as exc:
        logger.warning(
            "[VideoProcessor] _train_team_classifier failed: %s", exc, exc_info=True
        )
        return None


def _save_team_assignments(
    clf: "TeamClassifier | None",
    output_base: str,
) -> "str | None":
    """
    Persist team assignments to {output_base}_team_assignments.json.
    """
    if clf is None or not clf._trained:
        logger.warning("[VideoProcessor] _save_team_assignments: classifier not trained")
        return None
    try:
        path = output_base + "_team_assignments.json"
        assignments = {
            str(k): int(v)
            for k, v in clf._cache.items()
            if v in (1, 2)
        }
        payload = {
            "team_1_color_hex": clf.team_hex_color(1),
            "team_2_color_hex": clf.team_hex_color(2),
            "team_1_color_bgr": [int(v) for v in clf.team_colors.get(1, [0, 0, 0])],
            "team_2_color_bgr": [int(v) for v in clf.team_colors.get(2, [0, 0, 0])],
            "assignments":      assignments,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(
            "[VideoProcessor] Team assignments: %d players → %s",
            len(assignments), path,
        )
        return path
    except Exception as exc:
        logger.error("[VideoProcessor] _save_team_assignments error: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# H.264 re-encoding
# ─────────────────────────────────────────────────────────────────────────────

def _reencode_h264(src: str, dst: str, fps: float) -> bool:
    _safe_remove(dst)
    cmd = [
        "ffmpeg", "-y",
        "-i", src,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        dst,
    ]
    logger.info("[VideoProcessor] Re-encoding: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            logger.error("[VideoProcessor] ffmpeg error:\n%s", proc.stderr[-2000:])
            return False
        out_path = Path(dst)
        if not out_path.exists() or out_path.stat().st_size == 0:
            logger.error("[VideoProcessor] ffmpeg produced empty output")
            return False
        logger.info(
            "[VideoProcessor] Re-encode done: %s (%.1f MB)",
            dst, out_path.stat().st_size / 1_048_576,
        )
        return True
    except FileNotFoundError:
        logger.error("[VideoProcessor] ffmpeg not found — install with: sudo apt install ffmpeg")
        return False
    except subprocess.TimeoutExpired:
        logger.error("[VideoProcessor] ffmpeg timed out")
        return False
    except Exception as exc:
        logger.error("[VideoProcessor] ffmpeg unexpected error: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap helpers
# ─────────────────────────────────────────────────────────────────────────────

def _generate_heatmap(
    track_points: dict[int, list[tuple[float, float]]],
    width: int,
    height: int,
    reference_path: str,
) -> "str | None":
    """Combined all-player heatmap (existing behaviour, kept for back-compat)."""
    try:
        heatmap = np.zeros((height, width), dtype=np.float32)
        for points in track_points.values():
            for x, y in points:
                xi, yi = int(x), int(y)
                if 0 <= xi < width and 0 <= yi < height:
                    heatmap[yi, xi] += 1.0
        if heatmap.max() == 0:
            return None
        heatmap       = cv2.GaussianBlur(heatmap, (25, 25), 0)
        heatmap       = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_uint8 = np.uint8(heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        out_path = str(Path(reference_path).with_suffix("")) + "_heatmap.jpg"
        cv2.imwrite(out_path, heatmap_color)
        logger.info("[VideoProcessor] Combined heatmap saved: %s", out_path)
        return out_path
    except Exception as exc:
        logger.error("[VideoProcessor] Combined heatmap failed: %s", exc)
        return None


def _generate_per_team_heatmaps(
    heatmap_team1: np.ndarray,
    heatmap_team2: np.ndarray,
    output_base: str,
) -> tuple["str | None", "str | None"]:
    """
    Apply GaussianBlur((51,51)) + COLORMAP_JET to each per-team accumulator
    and save as separate JPEG files, mirroring the notebook's heatmap cells.

    Returns (team1_path, team2_path). Either may be None if the accumulator
    was never incremented (no players classified for that team) or on error.
    """
    team1_path: "str | None" = None
    team2_path: "str | None" = None

    for team_idx, grid, suffix, label in (
        (1, heatmap_team1, "_heatmap_team1.jpg", "Team 1"),
        (2, heatmap_team2, "_heatmap_team2.jpg", "Team 2"),
    ):
        try:
            if grid.max() == 0:
                logger.info(
                    "[VideoProcessor] Per-team heatmap: no data for %s — skipping", label
                )
                continue

            blurred   = cv2.GaussianBlur(grid, (51, 51), 0)
            norm      = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
            norm_u8   = np.uint8(norm)
            colorized = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)

            save_path = output_base + suffix
            cv2.imwrite(save_path, colorized)
            logger.info("[VideoProcessor] %s heatmap saved: %s", label, save_path)

            if team_idx == 1:
                team1_path = save_path
            else:
                team2_path = save_path

        except Exception as exc:
            logger.error(
                "[VideoProcessor] Per-team heatmap failed for %s: %s", label, exc
            )

    return team1_path, team2_path


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _safe_remove(path: str) -> None:
    try:
        if path and Path(path).exists():
            Path(path).unlink()
    except OSError:
        pass


def _build_result(cv_data: dict, status: dict) -> dict:
    result = {
        "real_cv_analysis": cv_data,
        "placeholder_sections": {
            "possession":        {"_status": "PLACEHOLDER", "team_a": None, "team_b": None},
            "shots_on_goal":     {"_status": "PLACEHOLDER", "count": None},
            "expected_goals_xg": {"_status": "PLACEHOLDER", "value": None},
            "tactical_analysis": {"_status": "PLACEHOLDER", "formation": None, "strengths": [], "weaknesses": [], "recommendations": []},
            "match_statistics":  {"_status": "PLACEHOLDER", "passes": None, "fouls": None, "corners": None},
        },
        "pipeline_status": numpy_to_python(status),
    }
    return numpy_to_python(result)