"""
heatmap.py
==========
Generates player-position heatmaps from tracking trajectories.

The notebook accumulated all (cx, cy) positions and then used OpenCV's
applyColorMap to overlay a gaussian-blurred density image on a pitch
background — exactly what this module reproduces.

Pipeline (from notebook pattern):
  1. Collect all (cx, cy) positions per team (or combined)
  2. Bin into a 2-D accumulation grid (frame_height × frame_width)
  3. Gaussian blur → normalise → apply JET colormap
  4. Blend with pitch background (or save standalone)
  5. Save as PNG alongside the processed video

TODO: If you want perspective-corrected heatmaps (positions in real metres
      on the pitch), apply a homography matrix from camera calibration before
      calling generate_*. Roboflow's sports repository has an example.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np  # type: ignore
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - cv2 optional
    cv2 = None

# Help static analyzers (VSCode/Pylance/mypy) recognize the cv2 symbols
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    import cv2  # type: ignore

from services.cv.tracker import TrackPoint

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pure-numpy grid accumulator (no OpenCV required)
# ---------------------------------------------------------------------------

def _accumulate_grid(
    points: List[TrackPoint],
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """
    Build a 2-D density grid from pixel-space track points.
    Returns float32 array of shape (frame_height, frame_width).
    """
    grid = np.zeros((frame_height, frame_width), dtype=np.float32)
    for p in points:
        xi = int(np.clip(p.cx, 0, frame_width  - 1))
        yi = int(np.clip(p.cy, 0, frame_height - 1))
        grid[yi, xi] += 1.0
    return grid


def _to_colour_heatmap(grid: np.ndarray) -> Optional[np.ndarray]:
    """
    Apply gaussian blur + normalise + JET colormap using OpenCV.
    Returns BGR uint8 image, or None if cv2 is unavailable.
    """
    try:
        import cv2
        from config import HEATMAP_ALPHA, HEATMAP_COLORMAP

        if grid.max() == 0:
            return None

        blurred  = cv2.GaussianBlur(grid, (0, 0), sigmaX=25, sigmaY=25)
        norm     = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
        norm_u8  = norm.astype(np.uint8)

        cmap = getattr(cv2, f"COLORMAP_{HEATMAP_COLORMAP.upper()}", cv2.COLORMAP_JET)
        coloured = cv2.applyColorMap(norm_u8, cmap)
        return coloured

    except ImportError:
        log.warning("[Heatmap] cv2 not available — cannot generate colour heatmap.")
        return None


def _to_normalised_matrix(grid: np.ndarray, rows: int = 10, cols: int = 16) -> List[List[float]]:
    """
    Downscale the full-resolution grid to a small rows×cols JSON-friendly matrix.
    Used by the frontend HeatmapPreview component regardless of cv2 availability.
    """
    if grid.max() == 0:
        return [[0.0] * cols for _ in range(rows)]

    try:
        from scipy.ndimage import zoom  # type: ignore
        scale_y = rows / grid.shape[0]
        scale_x = cols / grid.shape[1]
        small   = zoom(grid, (scale_y, scale_x))
    except ImportError:
        # Fallback: simple block averaging
        small = np.zeros((rows, cols), dtype=np.float32)
        bh = grid.shape[0] // rows
        bw = grid.shape[1] // cols
        for r in range(rows):
            for c in range(cols):
                block = grid[r*bh:(r+1)*bh, c*bw:(c+1)*bw]
                small[r, c] = block.mean()

    vmax = small.max()
    if vmax > 0:
        small /= vmax
    return small.round(4).tolist()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_heatmaps(
    player_points: List[TrackPoint],
    ball_points: List[TrackPoint],
    frame_width: int,
    frame_height: int,
    output_dir: Path,
) -> dict:
    """
    Generate and save heatmap images + return a metadata dict.

    Args:
        player_points : combined list of all player TrackPoints
        ball_points   : ball TrackPoints
        frame_width   : source video width in pixels
        frame_height  : source video height in pixels
        output_dir    : where to save PNG files

    Returns dict with keys:
        combined_heatmap_path, ball_heatmap_path, matrix (10×16 normalised)
    """
    from config import HEATMAP_GRID_ROWS, HEATMAP_GRID_COLS

    output_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "combined_heatmap_path": None,
        "ball_heatmap_path": None,
        "matrix": None,
    }

    # ── Player heatmap ────────────────────────────────────────────────────────
    if player_points:
        player_grid = _accumulate_grid(player_points, frame_width, frame_height)
        result["matrix"] = _to_normalised_matrix(player_grid, HEATMAP_GRID_ROWS, HEATMAP_GRID_COLS)

        colour = _to_colour_heatmap(player_grid)
        if colour is not None:
            try:
                import cv2
                path = output_dir / "heatmap_players.png"
                cv2.imwrite(str(path), colour)
                result["combined_heatmap_path"] = str(path)
                log.info("[Heatmap] Player heatmap saved to %s", path)
            except Exception as exc:
                log.error("[Heatmap] Failed to save player heatmap: %s", exc)
    else:
        log.info("[Heatmap] No player points — skipping player heatmap.")

    # ── Ball heatmap ──────────────────────────────────────────────────────────
    if ball_points:
        ball_grid  = _accumulate_grid(ball_points, frame_width, frame_height)
        colour     = _to_colour_heatmap(ball_grid)
        if colour is not None:
            try:
                import cv2
                path = output_dir / "heatmap_ball.png"
                cv2.imwrite(str(path), colour)
                result["ball_heatmap_path"] = str(path)
                log.info("[Heatmap] Ball heatmap saved to %s", path)
            except Exception as exc:
                log.error("[Heatmap] Failed to save ball heatmap: %s", exc)

    return result


def generate_placeholder_matrix() -> List[List[float]]:
    """
    Return a realistic-looking fake heatmap matrix for placeholder mode.
    Pitch centre and attacking thirds biased higher.
    """
    from config import HEATMAP_GRID_ROWS, HEATMAP_GRID_COLS
    import random

    rows, cols = HEATMAP_GRID_ROWS, HEATMAP_GRID_COLS
    matrix = []
    for r in range(rows):
        row = []
        for c in range(cols):
            centre_bias = 1.0 - (abs(r - rows // 2) / rows)
            attacking_bias = 0.2 if c > cols * 0.6 else 0.0
            val = round(random.uniform(0.05, 0.7) * centre_bias + attacking_bias, 4)
            val = min(val, 1.0)
            row.append(val)
        matrix.append(row)
    return matrix
