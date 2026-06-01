"""
video_processor.py — Fixed CV pipeline
"""

import cv2
import json
import logging
import numpy as np
from pathlib import Path
from typing import Any
from dataclasses import dataclass

import supervision as sv
from ultralytics import YOLO

from .notebook_adapter import get_annotator_bundle, numpy_to_python

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

    warning: str | None = None


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

    real = result.get("real_cv_analysis", {})
    status = result.get("pipeline_status", {})

    summary = {
        "total_frames_processed": real.get("frames_processed", 0),
        "total_detections": real.get("total_detections", 0),
        "max_players_in_frame": real.get("unique_tracks", 0),
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
        warning="; ".join(status.get("errors", [])) if status.get("errors") else None,
    )


def process_video(
    video_path: str,
    model_path: str,
    output_path: str,
    conf_threshold: float = 0.3,
) -> dict[str, Any]:

    status: dict[str, Any] = {
        "model_loaded": False,
        "video_opened": False,
        "frames_processed": 0,
        "annotation_ready": False,
        "output_written": False,
        "errors": [],
    }

    try:
        model = YOLO(model_path)
        status["model_loaded"] = True
        logger.info("[VideoProcessor] Model loaded: %s", model_path)
    except Exception as exc:
        status["errors"].append(f"Model load failed: {exc}")
        logger.error("[VideoProcessor] Model load failed: %s", exc)
        return _build_result({}, status)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        status["errors"].append(f"Cannot open video: {video_path}")
        return _build_result({}, status)

    status["video_opened"] = True

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
        "[VideoProcessor] Video: %dx%d @ %.1f fps, ~%d frames",
        width,
        height,
        fps,
        total_frames,
    )

    bundle = get_annotator_bundle()
    status["annotation_ready"] = getattr(bundle, "ready", False)

    tracker = sv.ByteTrack()

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path_obj), fourcc, fps, (width, height))

    if not out.isOpened():
        status["errors"].append(f"Cannot open VideoWriter: {output_path}")
        cap.release()
        return _build_result({}, status)

    all_track_points: dict[int, list[tuple[float, float]]] = {}
    total_detections = 0
    unique_track_ids: set[int] = set()

    detections_export: list[dict[str, Any]] = []
    tracking_export: list[dict[str, Any]] = []

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            results = model.predict(frame, conf=conf_threshold, verbose=False)
            raw = results[0]

            detections = sv.Detections.from_ultralytics(raw)
            detections = tracker.update_with_detections(detections)

            total_detections += len(detections)

            frame_detections = []

            for i, xyxy in enumerate(detections.xyxy):
                class_id = None
                confidence = None
                tracker_id = None

                if detections.class_id is not None:
                    class_id = int(detections.class_id[i])

                if detections.confidence is not None:
                    confidence = float(detections.confidence[i])

                if detections.tracker_id is not None:
                    tracker_id = int(detections.tracker_id[i])
                    unique_track_ids.add(tracker_id)

                    cx = float((xyxy[0] + xyxy[2]) / 2)
                    cy = float((xyxy[1] + xyxy[3]) / 2)

                    all_track_points.setdefault(tracker_id, []).append((cx, cy))

                    tracking_export.append(
                        {
                            "frame": int(frame_idx),
                            "track_id": tracker_id,
                            "center": [cx, cy],
                            "bbox": [float(v) for v in xyxy],
                        }
                    )

                frame_detections.append(
                    {
                        "class_id": class_id,
                        "confidence": confidence,
                        "tracker_id": tracker_id,
                        "bbox": [float(v) for v in xyxy],
                    }
                )

            detections_export.append(
                {
                    "frame": int(frame_idx),
                    "detections": frame_detections,
                }
            )

        except Exception as exc:
            logger.warning("[VideoProcessor] Frame %d inference error: %s", frame_idx, exc)
            detections = sv.Detections.empty()

        try:
            annotated = bundle.annotate_frame(frame, detections)
            if annotated is None:
                annotated = frame
        except Exception as exc:
            if frame_idx == 0:
                logger.warning("[VideoProcessor] Annotation disabled: %s", exc)
            annotated = frame

        if annotated.shape[1] == width and annotated.shape[0] == height:
            out.write(annotated)
        else:
            out.write(frame)

        frame_idx += 1

    cap.release()
    out.release()

    status["frames_processed"] = int(frame_idx)
    status["output_written"] = True

    logger.info(
        "[VideoProcessor] Done — %d frames, %d detections, %d tracks",
        frame_idx,
        total_detections,
        len(unique_track_ids),
    )

    heatmap_path = _generate_heatmap(all_track_points, width, height, str(output_path_obj))

    output_base = output_path_obj.with_suffix("")

    detections_json_path = str(output_base) + "_detections.json"
    tracking_json_path = str(output_base) + "_tracking.json"

    with open(detections_json_path, "w", encoding="utf-8") as f:
        json.dump(numpy_to_python(detections_export), f, ensure_ascii=False, indent=2)

    with open(tracking_json_path, "w", encoding="utf-8") as f:
        json.dump(numpy_to_python(tracking_export), f, ensure_ascii=False, indent=2)

    cv_data = {
        "frames_processed": int(frame_idx),
        "total_detections": int(total_detections),
        "unique_tracks": int(len(unique_track_ids)),
        "fps": float(round(fps, 2)),
        "resolution": {
            "width": int(width),
            "height": int(height),
        },
        "heatmap_path": heatmap_path,
        "output_video_path": str(output_path_obj),
        "detections_json_path": detections_json_path,
        "tracking_json_path": tracking_json_path,
    }

    return _build_result(cv_data, status)


def _generate_heatmap(
    track_points: dict[int, list[tuple[float, float]]],
    width: int,
    height: int,
    reference_path: str,
) -> str | None:
    try:
        heatmap = np.zeros((height, width), dtype=np.float32)

        for points in track_points.values():
            for x, y in points:
                xi, yi = int(x), int(y)
                if 0 <= xi < width and 0 <= yi < height:
                    heatmap[yi, xi] += 1.0

        if heatmap.max() == 0:
            return None

        heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_uint8 = np.uint8(heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        out_path = str(Path(reference_path).with_suffix("")) + "_heatmap.jpg"
        cv2.imwrite(out_path, heatmap_color)

        logger.info("[VideoProcessor] Heatmap saved: %s", out_path)
        return out_path

    except Exception as exc:
        logger.error("[VideoProcessor] Heatmap generation failed: %s", exc)
        return None


def _build_result(cv_data: dict, status: dict) -> dict:
    result = {
        "real_cv_analysis": cv_data,
        "placeholder_sections": {
            "possession": {
                "_status": "PLACEHOLDER — not implemented",
                "note": "Ball possession tracking requires ball detection + zone logic.",
                "team_a": None,
                "team_b": None,
            },
            "shots_on_goal": {
                "_status": "PLACEHOLDER — not implemented",
                "note": "Shot detection requires goal zone definition.",
                "count": None,
            },
            "expected_goals_xg": {
                "_status": "PLACEHOLDER — not implemented",
                "note": "xG requires shot angle / distance model.",
                "value": None,
            },
            "tactical_analysis": {
                "_status": "PLACEHOLDER — AI tactical analysis not connected",
                "note": "Requires formation recognition model.",
                "formation": None,
                "strengths": [],
                "weaknesses": [],
                "recommendations": [],
            },
            "match_statistics": {
                "_status": "PLACEHOLDER — statistics API not connected",
                "note": "Would require live or external data source.",
                "passes": None,
                "fouls": None,
                "corners": None,
            },
        },
        "pipeline_status": numpy_to_python(status),
    }

    return numpy_to_python(result)