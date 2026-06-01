"""
router_results.py — Fixed result endpoint

Drop this into backend/routers/ (or wherever your results router lives).

Key changes:
- Returns { real_cv_analysis, placeholder_sections, pipeline_status }
- Never mixes fake and real data in the same flat dict
- numpy types cast to Python native before JSON serialization
"""

import json
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from notebook_adapter import numpy_to_python   # import the helper

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/results", tags=["results"])


@router.get("/{job_id}")
async def get_results(job_id: str):
    """
    Return analysis results for *job_id*.

    The response shape is always:
    {
        "real_cv_analysis":     { ... actual detections / tracking data ... },
        "placeholder_sections": { ... sections not yet implemented ... },
        "pipeline_status":      { ... per-stage success flags ... }
    }
    """
    # ── Load stored result (adjust path to match your storage strategy) ────────
    result_path = Path(f"results/{job_id}.json")
    if not result_path.exists():
        raise HTTPException(status_code=404, detail=f"No result found for job {job_id}")

    try:
        with open(result_path) as f:
            raw = json.load(f)
    except Exception as exc:
        logger.error("[Results] Failed to load %s: %s", result_path, exc)
        raise HTTPException(status_code=500, detail="Failed to read result file")

    # ── Ensure the result has the new structure ────────────────────────────────
    if "real_cv_analysis" not in raw:
        # Migrate old flat results to the new shape
        raw = _migrate_legacy_result(raw)

    # ── Safe JSON serialization (numpy types etc.) ─────────────────────────────
    return JSONResponse(content=numpy_to_python(raw))


def _migrate_legacy_result(old: dict) -> dict:
    """
    Convert an old flat result dict to the new structured shape.
    Moves known CV fields to real_cv_analysis; everything else becomes
    placeholder_sections with explicit status labels.
    """
    cv_fields = {
        "frames_processed", "total_detections", "unique_tracks",
        "fps", "resolution", "heatmap_path", "output_video_path",
    }
    cv_data = {k: old[k] for k in cv_fields if k in old}
    remaining = {k: v for k, v in old.items() if k not in cv_fields}

    return {
        "real_cv_analysis": cv_data,
        "placeholder_sections": {
            "_migration_note": "Migrated from legacy flat result. Fields below were in the old response.",
            "_status": "PLACEHOLDER — data origin unknown, may be demo values",
            **remaining,
        },
        "pipeline_status": {
            "migrated_from_legacy": True,
            "errors": [],
        },
    }
