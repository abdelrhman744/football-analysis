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
    stats_engine,
    heatmap_summary_service,
)
from services.cv import CV_MODEL_AVAILABLE

router      = APIRouter()
RESULTS_DIR = "results"
log         = logging.getLogger(__name__)


def generate_ai_summary(video_id, detection, tracking, heatmap, teams, stats, cv_warning) -> AISummary:
    """
    Generate the initial AI summary.

    Narrative text (tactical_analysis, strengths/weaknesses, recommendations)
    remains hand-authored placeholder copy — that's explicitly the next
    project phase (Football Analyst Chatbot / RAG over real match data), and
    the frontend marks this section with a PlaceholderBadge accordingly.

    The numeric figures quoted in performance_summary, however, now come from
    real CV-derived stats (services.stats_engine) instead of fabricated xG
    values, since stats.home_team.expected_goals is genuinely unavailable
    (no shot-detection model exists yet) and would otherwise render as "None".
    """
    model_note = "" if cv_warning is None else " (placeholder data — CV model not loaded)"

    poss_a = stats.home_team.possession_percentage or 0.0
    poss_b = stats.away_team.possession_percentage or 0.0
    attacks_a = stats.attacks.get("A", 0)
    attacks_b = stats.attacks.get("B", 0)
    sprints_a = stats.sprints.get("A", 0)
    sprints_b = stats.sprints.get("B", 0)

    return AISummary(
        video_id=video_id,
        tactical_analysis=(
            f"Team A deployed a possession-based approach{model_note}, recording "
            f"{sprints_a} tracked sprints against Team B's {sprints_b}. "
            "Tactical shape and pressing-trigger analysis will be available once "
            "the Football Analyst Chatbot (RAG phase) is integrated."
        ),
        performance_summary=(
            f"Team A controlled {poss_a:.1f}% of possession to Team B's {poss_b:.1f}%. "
            f"Team A registered {attacks_a} tracked attacking sequences versus "
            f"{attacks_b} for Team B."
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


def build_unified_output(result: FullAnalysisResult) -> dict:
    """
    Collapse FullAnalysisResult into the single structured object the spec
    calls for — the future source of truth for the frontend and chatbot RAG.

    {
      "match_stats": {...},
      "player_stats": {...},
      "team_stats": {...},
      "heatmaps": {...},
      "analysis_summary": {...}
    }
    """
    stats = result.match_stats
    heatmap = result.heatmap

    return {
        "video_id": result.video_id,
        "match_stats": {
            "possession": stats.possession if stats else {},
            "ball_recovery": stats.ball_recovery if stats else {},
            "passes": stats.passes if stats else {},
            "attacks": stats.attacks if stats else {},
            "ball_speed": stats.ball_speed if stats else {},
        } if stats else {},
        "player_stats": stats.per_player if stats else {},
        "team_stats": {
            "team_a": stats.home_team.model_dump() if stats else {},
            "team_b": stats.away_team.model_dump() if stats else {},
        },
        "heatmaps": {
            "team_a_heatmap_url": heatmap.team_a_heatmap_url if heatmap else None,
            "team_b_heatmap_url": heatmap.team_b_heatmap_url if heatmap else None,
            "heatmap_matrix_a": stats.heatmap_matrix_a if stats else [],
            "heatmap_matrix_b": stats.heatmap_matrix_b if stats else [],
            "player_summaries": [s.model_dump() for s in heatmap.player_summaries] if heatmap else [],
        },
        "analysis_summary": result.ai_summary.model_dump() if result.ai_summary else {},
    }


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

    # Evict any stale cached stats from a previous run on this video_id so
    # this run's numbers reflect the latest tracking data, not a prior cache hit.
    match_stats_service.clear_cache(video_id)

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

        # ── 3b. Heatmap text summaries (NEW) ──────────────────────────────────
        # Reuses the same normalized tracking entries the stats engine loads,
        # so there's no duplicated parsing of the raw CV tracking JSON.
        try:
            raw_tracking = stats_engine.load_tracking(video_id)
            player_summaries = heatmap_summary_service.generate_player_heatmap_summaries(raw_tracking)
            heatmap.player_summaries = player_summaries
        except Exception as exc:  # non-critical — degrade gracefully
            log.warning("[analysis] Heatmap summary generation failed for %s: %s", video_id, exc)

        # ── 4. Team classification ────────────────────────────────────────────
        teams = team_classification_service.classify_teams(video_id, detection)

        # ── 5. Match stats (real CV-derived analytics) ───────────────────────
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

        # ── NEW: persist the unified analysis output for the chatbot's RAG ───
        unified = build_unified_output(result)
        with open(os.path.join(result_dir, "unified_analysis.json"), "w") as f:
            json.dump(unified, f, indent=2, default=str)

        # ── NEW: persist heatmap summaries separately too (easy RAG ingestion)
        with open(os.path.join(result_dir, "heatmap_summaries.json"), "w") as f:
            json.dump(unified["heatmaps"]["player_summaries"], f, indent=2, default=str)

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


@router.get("/unified/{video_id}")
async def get_unified_analysis(video_id: str):
    """
    NEW: Serve the single structured analysis object — the source of truth
    for the frontend and the future Football Analyst Chatbot's RAG index.
    """
    unified_path = os.path.join(RESULTS_DIR, video_id, "unified_analysis.json")
    if not os.path.exists(unified_path):
        raise HTTPException(
            status_code=404,
            detail=f"No unified analysis found for video '{video_id}'. "
                   f"Run POST /api/analysis/start/{video_id} first.",
        )
    with open(unified_path) as f:
        return json.load(f)


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
