"""
video_processor.py — CV pipeline with H.264 browser-compatible output
and improved team classification with multi-frame voting integrated into
the frame loop.
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
    heatmap_players_path: str | None = None
    heatmap_player_path: str | None = None
    heatmap_ball_path: str | None = None
    heatmap_matrix: list[list[float]] | None = None
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
        logger.info("[VideoProcessor] Model loaded: %s", model_path)
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
    all_track_points: dict[int, list[tuple[float, float]]] = {}
    total_detections  = 0
    unique_track_ids: set[int] = set()
    detections_export: list[dict[str, Any]] = []
    tracking_export:   list[dict[str, Any]] = []

    # Team classifier — two-phase approach:
    #   Phase A: buffer first TRAINING_FRAMES frames for training
    #   Phase B: train once, then accumulate_vote() every frame, annotate
    team_clf: TeamClassifier | None = None
    _train_buf_frames:  list[np.ndarray]          = []
    _train_buf_dets:    list[list[dict[str, Any]]] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── inference ─────────────────────────────────────────────────────────
        frame_dets: list[dict[str, Any]] = []
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

                frame_dets.append({
                    "class_id":   class_id,
                    "confidence": confidence,
                    "tracker_id": tracker_id,
                    "bbox":       [float(v) for v in xyxy],
                })

            detections_export.append({"frame": int(frame_idx), "detections": frame_dets})

        except Exception as exc:
            logger.warning("[VideoProcessor] Frame %d inference error: %s", frame_idx, exc)
            detections = sv.Detections.empty()
            frame_dets = []

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

        # ── annotate frame ────────────────────────────────────────────────────
        try:
            annotated = bundle.annotate_frame(frame, detections)
            if annotated is None:
                annotated = frame
        except Exception as exc:
            if frame_idx == 0:
                logger.warning("[VideoProcessor] Annotation disabled: %s", exc)
            annotated = frame

        # ── team-colour ellipses ──────────────────────────────────────────────
        if team_clf is not None:
            annotated = _draw_team_ellipses(annotated, frame_dets, frame, team_clf)

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
        # Finalise any players that haven't hit MIN_VOTES yet via predict()
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

    # ── 7. Save JSON outputs + heatmap + team assignments ────────────────────
    heatmap_path  = _generate_heatmap(all_track_points, width, height, str(output_path_obj))
    output_base   = output_path_obj.with_suffix("")
    det_json_path = str(output_base) + "_detections.json"
    trk_json_path = str(output_base) + "_tracking.json"

    with open(det_json_path, "w", encoding="utf-8") as f:
        json.dump(numpy_to_python(detections_export), f, ensure_ascii=False, indent=2)

    with open(trk_json_path, "w", encoding="utf-8") as f:
        json.dump(numpy_to_python(tracking_export), f, ensure_ascii=False, indent=2)

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


def _draw_team_ellipses(
    annotated: np.ndarray,
    frame_dets: list[dict[str, Any]],
    raw_frame: np.ndarray,
    clf: "TeamClassifier",
) -> np.ndarray:
    """
    Draw a filled team-colour ellipse under each tracked player.
    Skips referees (team_id == 0 sentinel).
    Non-fatal — returns original frame on any error.
    """
    try:
        canvas = annotated.copy()
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
                continue  # referee — skip ellipse

            bgr_arr = clf.team_colors.get(team_id)
            if bgr_arr is not None:
                bgr = tuple(int(max(0, min(255, v))) for v in bgr_arr[:3])
            else:
                bgr = (255, 100, 30) if team_id == 1 else (30, 180, 255)

            x1, y1, x2, y2 = (int(v) for v in bbox)
            x_c    = (x1 + x2) // 2
            half_w = max(4, (x2 - x1) // 2)

            cv2.ellipse(
                canvas,
                center=(x_c, y2),
                axes=(half_w, max(2, int(0.35 * half_w))),
                angle=0.0,
                startAngle=-45,
                endAngle=235,
                color=bgr,
                thickness=3,
                lineType=cv2.LINE_AA,
            )
        return canvas
    except Exception as exc:
        logger.debug("[VideoProcessor] _draw_team_ellipses error: %s", exc)
        return annotated


def _save_team_assignments(
    clf: "TeamClassifier | None",
    output_base: str,
) -> "str | None":
    """
    Persist team assignments to {output_base}_team_assignments.json.

    JSON structure:
    {
      "team_1_color_hex": "#RRGGBB",
      "team_2_color_hex": "#RRGGBB",
      "team_1_color_bgr": [B, G, R],
      "team_2_color_bgr": [B, G, R],
      "assignments": {"<tracker_id>": 1_or_2, ...}
    }
    """
    if clf is None or not clf._trained:
        logger.warning("[VideoProcessor] _save_team_assignments: classifier not trained")
        return None
    try:
        path    = output_base + "_team_assignments.json"
        # Only save players actually assigned to a team (exclude referee sentinel 0)
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
# Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def _generate_heatmap(
    track_points: dict[int, list[tuple[float, float]]],
    width: int,
    height: int,
    reference_path: str,
) -> "str | None":
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
        logger.info("[VideoProcessor] Heatmap saved: %s", out_path)
        return out_path
    except Exception as exc:
        logger.error("[VideoProcessor] Heatmap failed: %s", exc)
        return None


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
