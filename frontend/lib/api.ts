/**
 * API helper — mirrors all FastAPI backend schemas.
 *
 * API_BASE_URL is read from NEXT_PUBLIC_API_URL (set in .env.local).
 * Default: http://127.0.0.1:8000
 *
 * All fetch calls go directly to the backend — no Next.js rewrite required.
 */

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

/**
 * Convert a relative backend path (e.g. "/results/abc/processed.mp4")
 * to a full URL the browser can load.
 * Already-absolute URLs (http/https) are returned unchanged.
 */
export function toMediaUrl(path?: string | null): string | null {
  if (!path) return null;
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${API_BASE_URL}${path}`;
}

/** @deprecated use API_BASE_URL */
export const BACKEND_BASE = API_BASE_URL;

// ─── Shared types ──────────────────────────────────────────────────────────

export interface VideoMetadata {
  video_id: string;
  filename: string;
  file_size: number;
  duration_seconds: number | null;
  resolution: string | null;
  fps: number | null;
  upload_timestamp: string;
  file_path: string;
}

export interface VideoUploadResponse {
  success: boolean;
  video_id: string;
  metadata: VideoMetadata;
  message: string;
}

export interface DetectionResult {
  video_id: string;
  total_frames_analyzed: number;
  players_detected: number;
  ball_detections: number;
  status: string;
}

export interface TrackingPoint {
  frame: number;
  x: number;
  y: number;
  timestamp: number;
}

export interface PlayerTrack {
  player_id: number;
  team: string | null;
  track_points: TrackingPoint[];
  total_distance_meters: number | null;
  average_speed_kmh: number | null;
  max_speed_kmh: number | null;
}

export interface TrackingResult {
  video_id: string;
  total_players_tracked: number;
  player_tracks: PlayerTrack[];
  ball_track: TrackingPoint[];
  status: string;
}

export interface PlayerHeatmapSummary {
  player_id: number;
  team: string | null;
  most_active_zone: string;
  zone_share_pct: number;
  summary: string;
}

export interface HeatmapData {
  video_id: string;
  team_a_heatmap_url: string | null;
  team_b_heatmap_url: string | null;
  ball_heatmap_url: string | null;
  heatmap_matrix: number[][] | null;
  /** NEW: per-player text summaries from heatmap_summary_service */
  player_summaries: PlayerHeatmapSummary[];
  status: string;
}

export interface TeamPlayer {
  player_id: number;
  team_label: string;
  confidence: number;
  dominant_color: string | null;
}

export interface TeamClassificationResult {
  video_id: string;
  team_a_label: string;
  team_b_label: string;
  team_a_players: TeamPlayer[];
  team_b_players: TeamPlayer[];
  team_a_color: string | null;
  team_b_color: string | null;
  status: string;
}

export interface TeamStats {
  team_name: string;
  possession_percentage: number | null;
  shots: number | null;
  shots_on_target: number | null;
  passes: number | null;
  pass_accuracy: number | null;
  fouls: number | null;
  corners: number | null;
  yellow_cards: number | null;
  red_cards: number | null;
  expected_goals: number | null;
  goals: number | null;
}

export interface MatchStatsResult {
  video_id?: string | null;
  match_id?: string | null;
  home_team: TeamStats;
  away_team: TeamStats;
  status: string;
  data_source: string;
  // NEW: real CV-derived analytics from services.stats_engine
  possession: Record<string, number>;
  ball_recovery: Record<string, number>;
  passes: Record<string, number>;
  passing_network: Record<string, number>;
  attacks: Record<string, number>;
  sprints: Record<string, number>;
  ball_speed: { avg?: number; max?: number };
  per_player: Record<string, any>;
  possession_zones: Record<string, any>;
  heatmap_matrix_a: number[][];
  heatmap_matrix_b: number[][];
}

export interface AISummary {
  video_id: string;
  tactical_analysis: string;
  performance_summary: string;
  team_a_strengths: string[];
  team_a_weaknesses: string[];
  team_b_strengths: string[];
  team_b_weaknesses: string[];
  recommendations: string[];
  coaching_suggestions: string[];
  status: string;
}

export interface FullAnalysisResult {
  video_id: string;
  status: string;
  detection: DetectionResult | null;
  tracking: TrackingResult | null;
  heatmap: HeatmapData | null;
  team_classification: TeamClassificationResult | null;
  match_stats: MatchStatsResult | null;
  ai_summary: AISummary | null;
  processing_time_seconds: number | null;
}

export interface AnalysisStartResponse {
  success: boolean;
  video_id: string;
  message: string;
  /** Relative path, e.g. /results/{id}/processed.mp4 — pass through toMediaUrl() */
  processed_video_url: string | null;
  /** Relative path, e.g. /results/{id}/processed_heatmap.jpg — pass through toMediaUrl() */
  heatmap_url: string | null;
  result: FullAnalysisResult;
}

export interface CvStatus {
  cv_model_available: boolean;
  weights_path: string;
  weights_exist: boolean;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResponse {
  answer: string;
  video_id: string;
  sources_used: string[];
}

// ─── Shared error helper ───────────────────────────────────────────────────

/**
 * Safely extract an error message from a Response whose body may or may not
 * be JSON. Never throws a secondary parse error.
 */
async function extractErrorMessage(res: Response, fallback: string): Promise<string> {
  try {
    const ct = res.headers.get("content-type") ?? "";
    if (ct.includes("application/json")) {
      const body = await res.json();
      return body?.detail ?? body?.error ?? fallback;
    }
    const text = await res.text();
    return text.slice(0, 200) || fallback;   // cap length for display
  } catch {
    return fallback;
  }
}

// ─── API functions ─────────────────────────────────────────────────────────

export async function uploadVideo(
  file: File,
  onProgress?: (pct: number) => void
): Promise<VideoUploadResponse> {
  return new Promise((resolve, reject) => {
    const fd = new FormData();
    fd.append("file", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE_URL}/api/video/upload`);
    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable && onProgress)
        onProgress(Math.round((e.loaded / e.total) * 100));
    });
    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch {
          reject(new Error("Upload succeeded but response was not JSON"));
        }
      } else {
        try {
          reject(new Error(JSON.parse(xhr.responseText).detail || "Upload failed"));
        } catch {
          reject(new Error(`Upload failed (HTTP ${xhr.status})`));
        }
      }
    });
    xhr.addEventListener("error", () => reject(new Error("Network error during upload")));
    xhr.send(fd);
  });
}

export async function startAnalysis(videoId: string): Promise<AnalysisStartResponse> {
  const res = await fetch(`${API_BASE_URL}/api/analysis/start/${videoId}`, {
    method: "POST",
  });
  if (!res.ok) {
    const msg = await extractErrorMessage(res, "Failed to start analysis");
    throw new Error(msg);
  }
  return res.json();
}

export async function getAnalysisResult(videoId: string): Promise<FullAnalysisResult> {
  const res = await fetch(`${API_BASE_URL}/api/analysis/result/${videoId}`);
  if (!res.ok) {
    const msg = await extractErrorMessage(res, "Failed to fetch result");
    throw new Error(msg);
  }
  return res.json();
}

/** NEW: fetches the single structured output object (match_stats, player_stats,
 * team_stats, heatmaps, analysis_summary) — source of truth for RAG/chatbot. */
export async function getUnifiedAnalysis(videoId: string): Promise<Record<string, any>> {
  const res = await fetch(`${API_BASE_URL}/api/analysis/unified/${videoId}`);
  if (!res.ok) {
    const msg = await extractErrorMessage(res, "Failed to fetch unified analysis");
    throw new Error(msg);
  }
  return res.json();
}

export async function getCvStatus(): Promise<CvStatus> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/analysis/cv-status`);
    if (!res.ok) return { cv_model_available: false, weights_path: "", weights_exist: false };
    return res.json();
  } catch {
    return { cv_model_available: false, weights_path: "", weights_exist: false };
  }
}

export async function sendChatMessage(
  videoId: string,
  question: string,
  history: ChatMessage[]
): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_id: videoId, question, conversation_history: history }),
  });
  if (!res.ok) {
    const msg = await extractErrorMessage(res, "Chat request failed");
    throw new Error(msg);
  }
  return res.json();
}

export function formatBytes(bytes: number): string {
  if (!bytes) return "0 B";
  const k = 1024, sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}
