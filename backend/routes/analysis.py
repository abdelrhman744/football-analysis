import json
import logging
import os
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from models.schemas import (
    FullAnalysisResult,
    AnalysisStatus,
    AISummary,
    AnalysisStartResponse,
)
from services import (
    video_service,
    cv_detection_service,
    tracking_service,
    heatmap_service,
    team_classification_service,
    match_stats_service,
)
from services.cv import CV_MODEL_AVAILABLE

router      = APIRouter()
RESULTS_DIR = "results"
log         = logging.getLogger(__name__)


def generate_ai_summary(video_id, detection, tracking, heatmap, teams, stats, cv_warning) -> AISummary:
    """
    Generate initial AI summary.
    TODO: Replace with real LLM call passing all collected data as context.
    """
    model_note = "" if cv_warning is None else " (placeholder data — CV model not loaded)"

    return AISummary(
        video_id=video_id,
        tactical_analysis=(
            f"Team A deployed a possession-based 4-3-3 formation{model_note}. "
            "Team B responded with a compact 4-4-2 mid-block. "
            "The central midfield battle was key, with Team A winning more second balls."
        ),
        performance_summary=(
            f"Team A controlled {stats.home_team.possession_percentage}% of possession. "
            f"Both teams created meaningful chances — Team A xG {stats.home_team.expected_goals} "
            f"vs Team B {stats.away_team.expected_goals}."
        ),
        team_a_strengths=[
            "Strong possession retention in midfield",
            "Effective wide play through the channels",
            "High defensive line creating offside traps",
        ],
        team_a_weaknesses=[
            "Vulnerable to counter-attacks behind the defensive line",
            "Set-piece defending needs improvement",
        ],
        team_b_strengths=[
            "Compact defensive shape difficult to break down",
            "Direct counter-attack threat with pace up front",
        ],
        team_b_weaknesses=[
            "Lack of creativity in the final third",
            "Low possession percentage limits attacking time",
        ],
        recommendations=[
            "Team A should vary tempo to disorganize Team B's block",
            "Team B could benefit from pressing triggers higher up the pitch",
            "Both teams should work on set-piece routines",
        ],
        coaching_suggestions=[
            "Team A Coach: Consider introducing a second striker to stretch Team B's backline",
            "Team B Coach: Adjust wide midfielders to track Team A's overlapping fullbacks",
        ],
        status=AnalysisStatus.COMPLETED,
    )


@router.post("/start/{video_id}")
async def start_analysis(video_id: str):
    """
    Start the full analysis pipeline for a given video_id.

    Always returns valid JSON — never a plain-text 500.
    """
    if not video_service.video_exists(video_id):
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found.")

    video_path = video_service.get_video_path(video_id)
    start_time = time.time()

    try:
        # ── 1. CV Detection (+ tracking + heatmap in one video pass) ─────────
        detection  = cv_detection_service.detect_players_and_ball(video_id, video_path)

        cv_out     = getattr(detection, "_cv_output", None)
        cv_warning = getattr(cv_out, "warning", None) if cv_out else \
                     "CV model files not found — using placeholder results."

        # ── 2. Tracking ───────────────────────────────────────────────────────
        tracking = tracking_service.track_objects(video_id, detection)
        tracking._cv_output = cv_out  # type: ignore[attr-defined]

        # ── 3. Heatmap ────────────────────────────────────────────────────────
        heatmap = heatmap_service.generate_heatmap(video_id, tracking)

        # ── 4. Team classification ────────────────────────────────────────────
        teams = team_classification_service.classify_teams(video_id, detection)

        # ── 5. Match stats ────────────────────────────────────────────────────
        stats = await match_stats_service.fetch_match_stats(video_id)

        # ── 6. AI summary ─────────────────────────────────────────────────────
        summary = generate_ai_summary(
            video_id, detection, tracking, heatmap, teams, stats, cv_warning
        )

        processing_time = round(time.time() - start_time, 2)

        result = FullAnalysisResult(
            video_id=video_id,
            status=AnalysisStatus.COMPLETED,
            detection=detection,
            tracking=tracking,
            heatmap=heatmap,
            team_classification=teams,
            match_stats=stats,
            ai_summary=summary,
            processing_time_seconds=processing_time,
        )

        # ── Persist result ────────────────────────────────────────────────────
        result_dir = os.path.join(RESULTS_DIR, video_id)
        os.makedirs(result_dir, exist_ok=True)

        # model_dump() returns plain Python dicts/lists — safe to json.dump
        result_dict = result.model_dump()

        if cv_out:
            result_dict["cv_meta"] = {
                "used_real_model":      bool(getattr(cv_out, "used_real_model", False)),
                "processed_video_path": str(getattr(cv_out, "processed_video_path", "") or ""),
                "heatmap_player_path":  str(getattr(cv_out, "heatmap_player_path",  "") or ""),
                "heatmap_ball_path":    str(getattr(cv_out, "heatmap_ball_path",     "") or ""),
                "warning":              getattr(cv_out, "warning", None),
                "summary":              getattr(cv_out, "summary", {}),
            }

        with open(os.path.join(result_dir, "analysis.json"), "w") as f:
            json.dump(result_dict, f, indent=2, default=str)

        # ── Build URLs for frontend ───────────────────────────────────────────
        processed_video_url = f"/results/{video_id}/processed.mp4"
        heatmap_url         = f"/results/{video_id}/processed_heatmap.jpg"

        # Confirm files actually exist (don't return a broken URL)
        if not Path(f"{RESULTS_DIR}/{video_id}/processed.mp4").exists():
            processed_video_url = None
        if not Path(f"{RESULTS_DIR}/{video_id}/processed_heatmap.jpg").exists():
            heatmap_url = None

        model_status = "Real CV model active." if CV_MODEL_AVAILABLE else \
                       (cv_warning or "Placeholder mode.")

        response_payload = {
            "success":             True,
            "video_id":            video_id,
            "message":             f"Analysis completed in {processing_time}s. {model_status}",
            "processed_video_url": processed_video_url,
            "heatmap_url":         heatmap_url,
            "result":              result_dict,
        }

        return JSONResponse(content=response_payload)

    except Exception as exc:
        # Log the full traceback server-side, return structured JSON to client
        log.exception("[analysis] Pipeline crashed for video_id=%s: %s", video_id, exc)
        return JSONResponse(
            status_code=500,
            content={
                "success":   False,
                "video_id":  video_id,
                "error":     str(exc),
                "detail":    "Analysis pipeline failed. Check server logs for traceback.",
            },
        )


@router.get("/result/{video_id}", response_model=FullAnalysisResult)
async def get_analysis_result(video_id: str):
    """Retrieve stored analysis results for a video."""
    result_path = os.path.join(RESULTS_DIR, video_id, "analysis.json")
    if not os.path.exists(result_path):
        raise HTTPException(
            status_code=404,
            detail=f"No analysis results found for video '{video_id}'. "
                   f"Run POST /api/analysis/start/{video_id} first.",
        )

    with open(result_path) as f:
        data = json.load(f)

    # Remove cv_meta before passing to schema (not part of FullAnalysisResult)
    data.pop("cv_meta", None)
    return FullAnalysisResult(**data)


@router.get("/cv-status")
async def cv_status():
    """Returns whether the real CV model is loaded."""
    from config import MODEL_WEIGHTS_PATH
    from pathlib import Path
    return {
        "cv_model_available": CV_MODEL_AVAILABLE,
        "weights_path": MODEL_WEIGHTS_PATH,
        "weights_exist": Path(MODEL_WEIGHTS_PATH).exists(),
    }
