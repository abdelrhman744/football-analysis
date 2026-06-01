"""
detector.py
===========
YOLOv8-based player and ball detector.

This module is the direct port of the model-loading and inference cells
from the Kaggle notebook (youssefabdelshafy45/notebookfb2feab537).
The notebook used:
  - ultralytics YOLOv8 for detection
  - supervision for annotation/result parsing
  - classes: player (0), ball (1), [referee (2), goalkeeper (3)]

TODO: Drop your trained .pt file into backend/models_weights/ and set
      MODEL_WEIGHTS_PATH in config.py (or the env var MODEL_WEIGHTS_PATH).
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes (used whether or not the real model is loaded)
# ---------------------------------------------------------------------------

@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass
class Detection:
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str
    tracker_id: Optional[int] = None


@dataclass
class FrameDetections:
    frame_index: int
    timestamp: float                        # seconds
    players: List[Detection] = field(default_factory=list)
    ball: Optional[Detection] = None
    referees: List[Detection] = field(default_factory=list)

    @property
    def all_detections(self) -> List[Detection]:
        out = list(self.players) + list(self.referees)
        if self.ball:
            out.append(self.ball)
        return out


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class FootballDetector:
    """
    Wraps a YOLOv8 model for football-specific detection.

    Usage:
        detector = FootballDetector()
        if detector.is_ready:
            results = detector.detect_frame(bgr_frame, frame_index=0, fps=25)
    """

    def __init__(self) -> None:
        self._model = None
        self._ready = False
        self._try_load()

    def _try_load(self) -> None:
        """
        Attempt to load model weights. Silently falls back to placeholder
        mode if ultralytics is not installed or weights are missing.
        """
        try:
            from ultralytics import YOLO
            from config import MODEL_WEIGHTS_PATH, CLASS_PLAYER, CLASS_BALL

            weights = Path(MODEL_WEIGHTS_PATH)
            if not weights.exists():
                log.warning(
                    "[Detector] Weight file not found: %s — placeholder mode active.",
                    weights,
                )
                return

            log.info("[Detector] Loading YOLO weights from %s", weights)
            self._model = YOLO(str(weights))
            self._class_player   = CLASS_PLAYER
            self._class_ball     = CLASS_BALL

            # Warm-up: run a dummy inference to load CUDA/TensorRT kernels
            import numpy as np
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model(dummy, verbose=False)

            self._ready = True
            log.info("[Detector] Model ready.")

        except ImportError:
            log.warning("[Detector] ultralytics not installed — placeholder mode.")
        except Exception as exc:
            log.error("[Detector] Failed to load model: %s", exc)

    @property
    def is_ready(self) -> bool:
        return self._ready

    def detect_frame(
        self,
        frame,          # np.ndarray BGR
        frame_index: int = 0,
        fps: float = 25.0,
    ) -> FrameDetections:
        """
        Run YOLOv8 inference on a single BGR frame.

        TODO: The notebook used model.predict() with imgsz=1280 and
              conf/iou thresholds. Adjust YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD,
              and YOLO_IMG_SIZE in config.py to match your training setup.

        Returns:
            FrameDetections with separated player, ball, and referee lists.
        """
        from config import (
            YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD, YOLO_IMG_SIZE,
            CLASS_PLAYER, CLASS_BALL, CLASS_REFEREE, CLASS_GOALKEEPER,
        )

        result = FrameDetections(
            frame_index=frame_index,
            timestamp=round(frame_index / fps, 3),
        )

        if not self._ready or self._model is None:
            return result

        try:
            # ── Real inference ────────────────────────────────────────────────
            # This mirrors the notebook cell:
            #   results = model.predict(frame, conf=0.35, iou=0.45, imgsz=1280)
            preds = self._model.predict(
                frame,
                conf=YOLO_CONF_THRESHOLD,
                iou=YOLO_IOU_THRESHOLD,
                imgsz=YOLO_IMG_SIZE,
                verbose=False,
            )

            for pred in preds:
                boxes = pred.boxes
                if boxes is None:
                    continue

                xyxy   = boxes.xyxy.cpu().numpy()
                confs  = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)

                for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, cls_ids):
                    cls_name = self._model.names.get(cls_id, str(cls_id))
                    det = Detection(
                        bbox=BBox(float(x1), float(y1), float(x2), float(y2)),
                        confidence=float(conf),
                        class_id=cls_id,
                        class_name=cls_name,
                    )

                    if cls_id == CLASS_BALL:
                        # Keep only the highest-confidence ball detection
                        if result.ball is None or conf > result.ball.confidence:
                            result.ball = det
                    elif cls_id in (CLASS_REFEREE,):
                        result.referees.append(det)
                    else:
                        # player and goalkeeper both go to players list
                        result.players.append(det)

        except Exception as exc:
            log.error("[Detector] Inference error on frame %d: %s", frame_index, exc)

        return result


# Module-level singleton — imported by video_processor and tracking
_detector: Optional[FootballDetector] = None


def get_detector() -> FootballDetector:
    """Return (or lazily create) the module-level detector singleton."""
    global _detector
    if _detector is None:
        _detector = FootballDetector()
    return _detector
