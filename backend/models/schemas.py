"""
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class AnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ─── Video Schemas ───────────────────────────────────────────────────────────

class VideoMetadata(BaseModel):
    video_id: str
    filename: str
    file_size: int
    duration_seconds: Optional[float] = None
    resolution: Optional[str] = None
    fps: Optional[float] = None
    upload_timestamp: str
    file_path: str


class VideoUploadResponse(BaseModel):
    success: bool
    video_id: str
    metadata: VideoMetadata
    message: str


# ─── Detection Schemas ────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    confidence: float


class DetectedPlayer(BaseModel):
    player_id: int
    bounding_box: BoundingBox
    team: Optional[str] = None
    frame_number: int


class DetectedBall(BaseModel):
    bounding_box: BoundingBox
    frame_number: int


class DetectionResult(BaseModel):
    video_id: str
    total_frames_analyzed: int
    players_detected: int
    ball_detections: int
    players: List[DetectedPlayer] = []
    balls: List[DetectedBall] = []
    status: AnalysisStatus
    # TODO: Add real detection fields when integrating CV model


# ─── Tracking Schemas ─────────────────────────────────────────────────────────

class TrackingPoint(BaseModel):
    frame: int
    x: float
    y: float
    timestamp: float


class PlayerTrack(BaseModel):
    player_id: int
    team: Optional[str] = None
    track_points: List[TrackingPoint] = []
    total_distance_meters: Optional[float] = None
    average_speed_kmh: Optional[float] = None
    max_speed_kmh: Optional[float] = None


class TrackingResult(BaseModel):
    video_id: str
    total_players_tracked: int
    player_tracks: List[PlayerTrack] = []
    ball_track: List[TrackingPoint] = []
    status: AnalysisStatus
    # TODO: Add real tracking fields when integrating tracking model


# ─── Heatmap Schemas ──────────────────────────────────────────────────────────

class HeatmapData(BaseModel):
    video_id: str
    team_a_heatmap_url: Optional[str] = None
    team_b_heatmap_url: Optional[str] = None
    ball_heatmap_url: Optional[str] = None
    heatmap_matrix: Optional[List[List[float]]] = None
    status: AnalysisStatus
    # TODO: Add real heatmap fields when integrating heatmap generation


# ─── Team Classification Schemas ─────────────────────────────────────────────

class TeamPlayer(BaseModel):
    player_id: int
    team_label: str
    confidence: float
    dominant_color: Optional[str] = None


class TeamClassificationResult(BaseModel):
    video_id: str
    team_a_label: str
    team_b_label: str
    team_a_players: List[TeamPlayer] = []
    team_b_players: List[TeamPlayer] = []
    team_a_color: Optional[str] = None
    team_b_color: Optional[str] = None
    status: AnalysisStatus
    # TODO: Add real classification fields when integrating team classification model


# ─── Match Stats Schemas ──────────────────────────────────────────────────────

class TeamStats(BaseModel):
    team_name: str
    possession_percentage: Optional[float] = None
    shots: Optional[int] = None
    shots_on_target: Optional[int] = None
    passes: Optional[int] = None
    pass_accuracy: Optional[float] = None
    fouls: Optional[int] = None
    corners: Optional[int] = None
    yellow_cards: Optional[int] = None
    red_cards: Optional[int] = None
    expected_goals: Optional[float] = None
    goals: Optional[int] = None


class MatchStatsResult(BaseModel):
    match_id: Optional[str] = None
    video_id: Optional[str] = None
    home_team: TeamStats
    away_team: TeamStats
    status: AnalysisStatus
    data_source: str = "placeholder"
    # TODO: Replace with real external football API data


# ─── AI Summary Schemas ───────────────────────────────────────────────────────

class AISummary(BaseModel):
    video_id: str
    tactical_analysis: str
    performance_summary: str
    team_a_strengths: List[str] = []
    team_a_weaknesses: List[str] = []
    team_b_strengths: List[str] = []
    team_b_weaknesses: List[str] = []
    recommendations: List[str] = []
    coaching_suggestions: List[str] = []
    status: AnalysisStatus
    # TODO: Generate from real AI/LLM model with full match context


# ─── Full Analysis Result ─────────────────────────────────────────────────────

class FullAnalysisResult(BaseModel):
    video_id: str
    status: AnalysisStatus
    detection: Optional[DetectionResult] = None
    tracking: Optional[TrackingResult] = None
    heatmap: Optional[HeatmapData] = None
    team_classification: Optional[TeamClassificationResult] = None
    match_stats: Optional[MatchStatsResult] = None
    ai_summary: Optional[AISummary] = None
    processing_time_seconds: Optional[float] = None


class AnalysisStartResponse(BaseModel):
    success: bool
    video_id: str
    message: str
    result: FullAnalysisResult


# ─── Chatbot Schemas ──────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    video_id: str
    question: str
    conversation_history: List[ChatMessage] = []


class ChatResponse(BaseModel):
    answer: str
    video_id: str
    sources_used: List[str] = []
    # TODO: Add references to specific match events when real data is integrated
