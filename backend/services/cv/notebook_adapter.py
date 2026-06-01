"""
notebook_adapter.py — Fixed for supervision 0.28+

Changes from broken version:
- Replaced BoundingBoxAnnotator → sv.BoxAnnotator (current API)
- Replaced Color.from_rgb() → sv.Color.from_rgb_tuple() (current API)
- Added defensive try/except around annotator init (logs ONCE, not per frame)
- annotate_frame() falls back to returning original frame on any error
- numpy int64/float64 values are cast to Python native types before JSON serialization
"""

import logging
import numpy as np
import cv2
from typing import Optional

logger = logging.getLogger(__name__)

# ── Try to import supervision ──────────────────────────────────────────────────
try:
    import supervision as sv
    _SV_AVAILABLE = True
except ImportError:
    _SV_AVAILABLE = False
    logger.warning("[NotebookAdapter] supervision not installed — annotation disabled")


# ── Annotator initialization ───────────────────────────────────────────────────

class AnnotatorBundle:
    """
    Holds the supervision annotators and exposes annotate_frame().
    Falls back gracefully if any annotator fails to initialize or annotate.
    """

    def __init__(self):
        self._ready = False
        self._box_annotator: Optional[object] = None
        self._label_annotator: Optional[object] = None
        self._trace_annotator: Optional[object] = None

        if not _SV_AVAILABLE:
            logger.warning("[NotebookAdapter] Skipping annotator init — supervision missing")
            return

        try:
            # supervision 0.25+ uses BoxAnnotator (not BoundingBoxAnnotator)
            # Color.from_rgb_tuple() replaces the removed Color.from_rgb()
            self._box_annotator = sv.BoxAnnotator(
                color=sv.ColorPalette.DEFAULT,
                thickness=2,
            )
            self._label_annotator = sv.LabelAnnotator(
                color=sv.ColorPalette.DEFAULT,
                text_color=sv.Color.WHITE,
                text_scale=0.5,
                text_thickness=1,
                text_padding=5,
            )
            self._trace_annotator = sv.TraceAnnotator(
                color=sv.ColorPalette.DEFAULT,
                thickness=2,
            )
            self._ready = True
            logger.info("[NotebookAdapter] Annotators initialized successfully (supervision %s)", sv.__version__)
        except Exception as exc:
            # Log ONCE — not per frame
            logger.error("[NotebookAdapter] Could not init supervision annotators: %s", exc)
            self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def annotate_frame(self, frame: np.ndarray, detections) -> np.ndarray:
        """
        Draw bounding boxes, labels, and traces on *frame*.

        Parameters
        ----------
        frame      : BGR numpy array (H, W, 3)
        detections : sv.Detections object (may be empty)

        Returns
        -------
        Annotated BGR frame.  If anything goes wrong the original frame is
        returned unchanged — the pipeline must never crash here.
        """
        if frame is None:
            logger.debug("[NotebookAdapter] annotate_frame received None frame")
            return frame

        if not self._ready:
            # Already logged at init time — stay silent here
            return frame

        try:
            annotated = frame.copy()

            # Boxes
            if self._box_annotator is not None:
                annotated = self._box_annotator.annotate(
                    scene=annotated, detections=detections
                )

            # Labels (tracker ID or class name)
            if self._label_annotator is not None:
                labels = _build_labels(detections)
                annotated = self._label_annotator.annotate(
                    scene=annotated, detections=detections, labels=labels
                )

            # Motion traces (only when tracker IDs are present)
            if self._trace_annotator is not None and detections.tracker_id is not None:
                annotated = self._trace_annotator.annotate(
                    scene=annotated, detections=detections
                )

            return annotated

        except Exception as exc:
            logger.error("[NotebookAdapter] annotate_frame error: %s", exc)
            return frame  # safe fallback — never crash the pipeline


# ── Internal helpers ───────────────────────────────────────────────────────────

def _build_labels(detections) -> list[str]:
    """Build per-detection label strings."""
    labels = []
    n = len(detections)
    for i in range(n):
        parts = []
        # Tracker ID (e.g. ByteTrack)
        if detections.tracker_id is not None and i < len(detections.tracker_id):
            parts.append(f"#{int(detections.tracker_id[i])}")
        # Confidence
        if detections.confidence is not None and i < len(detections.confidence):
            parts.append(f"{float(detections.confidence[i]):.2f}")
        labels.append(" ".join(parts) if parts else "")
    return labels


# ── JSON serialization helper ──────────────────────────────────────────────────

def numpy_to_python(obj):
    """
    Recursively convert numpy scalars / arrays to plain Python types so that
    json.dumps() never raises 'Object of type int64 is not JSON serializable'.

    Usage:
        import json
        from notebook_adapter import numpy_to_python
        json.dumps(numpy_to_python(my_dict))
    """
    if isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [numpy_to_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ── Singleton ──────────────────────────────────────────────────────────────────

_annotator_bundle: Optional[AnnotatorBundle] = None


def get_annotator_bundle() -> AnnotatorBundle:
    """Return the shared AnnotatorBundle, initializing it on first call."""
    global _annotator_bundle
    if _annotator_bundle is None:
        _annotator_bundle = AnnotatorBundle()
    return _annotator_bundle


def annotate_frame(frame: np.ndarray, detections) -> np.ndarray:
    """Module-level convenience wrapper around get_annotator_bundle().annotate_frame()."""
    return get_annotator_bundle().annotate_frame(frame, detections)
