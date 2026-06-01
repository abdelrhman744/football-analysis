"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import Image from "next/image";
import {
  Users, Target, Activity, BarChart2, Zap, Upload,
  Eye, GitBranch, Clock, Video, CheckCircle2, AlertCircle,
  Cpu, Info
} from "lucide-react";
import StatsCard from "@/components/StatsCard";
import HeatmapPreview from "@/components/HeatmapPreview";
import RecommendationCard from "@/components/RecommendationCard";
import {
  getAnalysisResult, getCvStatus, FullAnalysisResult,
  CvStatus, formatBytes, BACKEND_BASE
} from "@/lib/api";

// ── Small helper components ────────────────────────────────────────────────

function ModelBadge({ status }: { status: CvStatus | null }) {
  if (!status) return null;
  const real = status.cv_model_available;
  return (
    <div className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium border ${
      real
        ? "bg-pitch-500/10 border-pitch-500/20 text-pitch-400"
        : "bg-yellow-500/10 border-yellow-500/20 text-yellow-400"
    }`}>
      <span className={`h-1.5 w-1.5 rounded-full ${real ? "bg-pitch-500" : "bg-yellow-400"} animate-pulse`} />
      {real ? "Real CV Model Active" : "Placeholder Mode"}
    </div>
  );
}

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="font-display text-lg font-600 text-white uppercase tracking-wider mb-5">
      {children}
    </h2>
  );
}

function PlaceholderBadge() {
  return (
    <span className="ml-2 inline-flex items-center gap-1 text-xs text-slate-600 border border-white/5 rounded-full px-2 py-0.5">
      <Info className="h-3 w-3" /> placeholder
    </span>
  );
}

// ── Processed Video Player ─────────────────────────────────────────────────

function VideoPlayer({ videoId, hasMeta }: { videoId: string; hasMeta: boolean }) {
  const src = `${BACKEND_BASE}/results/${videoId}/processed.mp4`;
  return (
    <div className="glass-card rounded-xl p-6 mb-6">
      <SectionHeading>Processed Video</SectionHeading>
      <div className="aspect-video rounded-lg overflow-hidden bg-black relative">
        <video
          controls
          className="w-full h-full object-contain"
          preload="metadata"
          src={src}
          onError={(e) => {
            // Hide video element and show placeholder when real file missing
            (e.target as HTMLVideoElement).style.display = "none";
          }}
        />
        {/* Fallback overlay */}
        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none bg-navy-900/80">
          <Video className="h-12 w-12 text-slate-700 mb-3" />
          <p className="text-sm text-slate-600">
            Annotated video will appear here after the real CV model runs.
          </p>
        </div>
      </div>
      <p className="text-xs text-slate-600 mt-2">
        Annotated output saved to{" "}
        <span className="font-mono">backend/results/{videoId}/processed.mp4</span>
      </p>
    </div>
  );
}

// ── Detection Summary ──────────────────────────────────────────────────────

function DetectionSection({ detection, usedReal }: {
  detection: FullAnalysisResult["detection"];
  usedReal: boolean;
}) {
  if (!detection) return null;
  return (
    <div className="glass-card rounded-xl p-6 mb-6">
      <div className="flex items-center gap-2 mb-5">
        <SectionHeading>Detection Summary</SectionHeading>
        {!usedReal && <PlaceholderBadge />}
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <StatsCard title="Frames Analyzed" value={detection.total_frames_analyzed.toLocaleString()} icon={Eye} subtitle="CV model output" />
        <StatsCard title="Players Detected" value={detection.players_detected} icon={Users} accent subtitle="Unique player IDs" />
        <StatsCard title="Ball Frames" value={detection.ball_detections.toLocaleString()} icon={Target} subtitle="Frames with ball" />
        <StatsCard title="Detection Status" value={detection.status === "completed" ? "Done" : detection.status} icon={CheckCircle2} accent={detection.status === "completed"} />
      </div>
    </div>
  );
}

// ── Tracking Summary ───────────────────────────────────────────────────────

function TrackingSection({ tracking, usedReal }: {
  tracking: FullAnalysisResult["tracking"];
  usedReal: boolean;
}) {
  if (!tracking) return null;
  return (
    <div className="glass-card rounded-xl p-6 mb-6">
      <div className="flex items-center gap-2 mb-5">
        <SectionHeading>Tracking Result</SectionHeading>
        {!usedReal && <PlaceholderBadge />}
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-5">
        <StatsCard title="Players Tracked" value={tracking.total_players_tracked} icon={GitBranch} accent />
        <StatsCard title="Ball Positions" value={tracking.ball_track.length} icon={Activity} subtitle="Recorded positions" />
        <StatsCard title="Tracking Status" value={tracking.status === "completed" ? "Done" : tracking.status} icon={CheckCircle2} accent={tracking.status === "completed"} />
        <StatsCard title="Teams" value="2" icon={Users} subtitle="A & B classified" />
      </div>

      {tracking.player_tracks.length > 0 && (
        <>
          <h3 className="text-sm font-semibold text-slate-400 mb-3 uppercase tracking-wider">
            Top Players by Distance
          </h3>
          <div className="space-y-2">
            {tracking.player_tracks
              .filter((t) => t.total_distance_meters != null)
              .sort((a, b) => (b.total_distance_meters ?? 0) - (a.total_distance_meters ?? 0))
              .slice(0, 5)
              .map((t) => (
                <div key={t.player_id} className="flex items-center gap-3 py-2 border-b border-white/4">
                  <span className="text-xs font-mono text-slate-500 w-12">#{t.player_id}</span>
                  <span className={`text-xs px-2 py-0.5 rounded-full ${
                    t.team === "Team A"
                      ? "bg-white/10 text-white"
                      : "bg-blue-900/40 text-blue-300"
                  }`}>{t.team ?? "—"}</span>
                  <div className="flex-1 h-1.5 rounded-full bg-white/5">
                    <div
                      className="h-full rounded-full bg-pitch-500/60"
                      style={{ width: `${Math.min(100, ((t.total_distance_meters ?? 0) / 12000) * 100)}%` }}
                    />
                  </div>
                  <span className="text-xs text-slate-400 w-20 text-right">
                    {t.total_distance_meters != null
                      ? `${(t.total_distance_meters / 1000).toFixed(2)} km`
                      : "—"}
                  </span>
                  <span className="text-xs text-slate-600 w-20 text-right">
                    {t.max_speed_kmh != null ? `${t.max_speed_kmh} km/h` : "—"}
                  </span>
                </div>
              ))}
          </div>
        </>
      )}
    </div>
  );
}

// ── Main Dashboard ─────────────────────────────────────────────────────────

export default function DashboardPage() {
  const [result, setResult]     = useState<FullAnalysisResult | null>(null);
  const [meta, setMeta]         = useState<Record<string, string> | null>(null);
  const [cvStatus, setCvStatus] = useState<CvStatus | null>(null);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState<string | null>(null);

  useEffect(() => {
    const videoId  = localStorage.getItem("lastVideoId");
    const metaRaw  = localStorage.getItem("lastVideoMeta");
    if (metaRaw) setMeta(JSON.parse(metaRaw));

    Promise.all([
      videoId ? getAnalysisResult(videoId).catch((e) => { setError(e.message); return null; }) : Promise.resolve(null),
      getCvStatus(),
    ]).then(([r, cv]) => {
      setResult(r);
      setCvStatus(cv);
    }).finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="min-h-[calc(100vh-64px)] flex items-center justify-center">
        <Activity className="h-8 w-8 text-pitch-500 animate-pulse" />
      </div>
    );
  }

  if (!result) {
    return (
      <div className="min-h-[calc(100vh-64px)] flex items-center justify-center px-4">
        <div className="text-center max-w-md">
          <div className="h-16 w-16 rounded-2xl bg-white/3 border border-white/5 flex items-center justify-center mx-auto mb-4">
            <Upload className="h-8 w-8 text-slate-600" />
          </div>
          <h2 className="text-xl font-display font-600 text-white mb-2 uppercase">No Analysis Yet</h2>
          <p className="text-slate-500 text-sm mb-6">{error || "Upload a match video to begin."}</p>
          <Link href="/upload" className="inline-flex items-center gap-2 rounded-lg bg-pitch-500 px-6 py-3 text-sm font-semibold text-navy-950 hover:bg-pitch-400 transition-colors">
            Upload Match Video
          </Link>
        </div>
      </div>
    );
  }

  const { detection, tracking, heatmap, team_classification, match_stats, ai_summary, processing_time_seconds } = result;
  // Determine if real model was used (persisted in cv_meta, approximate via heatmap url presence)
  const usedReal = !!(heatmap?.team_a_heatmap_url);

  return (
    <div className="min-h-[calc(100vh-64px)] px-4 py-12">
      <div className="mx-auto max-w-7xl">

        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4 mb-8">
          <div>
            <h1 className="font-display text-4xl font-700 text-white uppercase tracking-wide">
              Match Dashboard
            </h1>
            <p className="text-slate-500 text-sm mt-1">
              Video: <span className="font-mono text-slate-400">{result.video_id.slice(0, 16)}…</span>
              {processing_time_seconds && (
                <span className="ml-3 inline-flex items-center gap-1 text-slate-600">
                  <Clock className="h-3 w-3" /> {processing_time_seconds}s
                </span>
              )}
            </p>
          </div>
          <div className="flex flex-col items-end gap-2">
            <ModelBadge status={cvStatus} />
            <Link href="/chat" className="flex items-center gap-2 rounded-lg bg-pitch-500/10 border border-pitch-500/20 px-4 py-2.5 text-sm font-medium text-pitch-400 hover:bg-pitch-500/15 transition-colors">
              <Zap className="h-4 w-4" /> Ask AI Analyst
            </Link>
          </div>
        </div>

        {/* CV model not loaded warning */}
        {!cvStatus?.cv_model_available && (
          <div className="glass-card rounded-xl px-5 py-4 mb-6 flex items-start gap-3 border border-yellow-500/15">
            <AlertCircle className="h-5 w-5 text-yellow-400 shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-yellow-400 mb-0.5">Placeholder Mode Active</p>
              <p className="text-xs text-slate-500">
                The real CV model weights are not loaded. Results shown are synthetic.
                Place your <span className="font-mono">player_ball_detector.pt</span> in{" "}
                <span className="font-mono">backend/models_weights/</span> and restart the server.
              </p>
            </div>
          </div>
        )}

        {/* Video info bar */}
        {meta && (
          <div className="glass-card rounded-xl p-5 mb-6 grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div><p className="text-xs text-slate-500 mb-1">Filename</p><p className="text-sm font-medium text-white truncate">{meta.filename}</p></div>
            <div><p className="text-xs text-slate-500 mb-1">File Size</p><p className="text-sm font-medium text-white">{formatBytes(Number(meta.file_size))}</p></div>
            <div><p className="text-xs text-slate-500 mb-1">Uploaded</p><p className="text-sm font-medium text-white">{new Date(meta.upload_timestamp).toLocaleString()}</p></div>
            <div><p className="text-xs text-slate-500 mb-1">Status</p><span className="inline-flex items-center gap-1.5 text-xs font-medium text-pitch-500"><span className="h-1.5 w-1.5 rounded-full bg-pitch-500 animate-pulse" />Complete</span></div>
          </div>
        )}

        {/* Processed video */}
        <VideoPlayer videoId={result.video_id} hasMeta={!!meta} />

        {/* Detection */}
        <DetectionSection detection={detection} usedReal={usedReal} />

        {/* Tracking */}
        <TrackingSection tracking={tracking} usedReal={usedReal} />

        {/* Heatmap */}
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-1">
            {!usedReal && (
              <div className="ml-auto">
                <PlaceholderBadge />
              </div>
            )}
          </div>
          <HeatmapPreview matrix={heatmap?.heatmap_matrix ?? null} />
          {heatmap?.team_a_heatmap_url && (
            <div className="mt-3 glass-card rounded-xl p-4">
              <p className="text-xs text-slate-500 mb-2">Real heatmap images from CV model:</p>
              <div className="grid grid-cols-2 gap-3">
                {heatmap.team_a_heatmap_url && (
                  <div>
                    <p className="text-xs text-slate-600 mb-1">Players</p>
                    <img
                      src={`${BACKEND_BASE}${heatmap.team_a_heatmap_url}`}
                      alt="Player heatmap"
                      className="w-full rounded-lg object-cover"
                    />
                  </div>
                )}
                {heatmap.ball_heatmap_url && (
                  <div>
                    <p className="text-xs text-slate-600 mb-1">Ball</p>
                    <img
                      src={`${BACKEND_BASE}${heatmap.ball_heatmap_url}`}
                      alt="Ball heatmap"
                      className="w-full rounded-lg object-cover"
                    />
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Team classification */}
        {team_classification && (
          <div className="glass-card rounded-xl p-6 mb-6">
            <div className="flex items-center gap-2 mb-5">
              <SectionHeading>Team Classification</SectionHeading>
              <PlaceholderBadge />
            </div>
            <div className="grid grid-cols-2 gap-4">
              {[
                { label: team_classification.team_a_label, color: team_classification.team_a_color ?? "#fff", count: team_classification.team_a_players?.length ?? 0 },
                { label: team_classification.team_b_label, color: team_classification.team_b_color ?? "#003366", count: team_classification.team_b_players?.length ?? 0 },
              ].map(({ label, color, count }) => (
                <div key={label} className="flex items-center gap-4 rounded-xl bg-white/3 p-4">
                  <div className="h-10 w-10 rounded-lg border border-white/10 shrink-0" style={{ backgroundColor: color }} />
                  <div>
                    <p className="font-semibold text-white">{label}</p>
                    <p className="text-sm text-slate-500">{count} players classified</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Match stats */}
        {match_stats && (
          <div className="glass-card rounded-xl p-6 mb-6">
            <div className="flex items-center gap-2 mb-5">
              <SectionHeading>Match Statistics</SectionHeading>
              <PlaceholderBadge />
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              {[match_stats.home_team, match_stats.away_team].map((team, ti) => (
                <div key={team.team_name}>
                  <div className="flex items-center gap-2 mb-4">
                    <div className="h-3 w-3 rounded-sm" style={{ backgroundColor: ti === 0 ? "#ffffff" : "#003366" }} />
                    <span className="font-semibold text-white">{team.team_name}</span>
                  </div>
                  {team.possession_percentage != null && (
                    <div className="mb-3">
                      <div className="flex justify-between text-xs text-slate-500 mb-1">
                        <span>Possession</span>
                        <span className="text-white">{team.possession_percentage}%</span>
                      </div>
                      <div className="h-1.5 rounded-full bg-white/5">
                        <div className="h-full rounded-full bg-pitch-500/70" style={{ width: `${team.possession_percentage}%` }} />
                      </div>
                    </div>
                  )}
                  <div className="grid grid-cols-3 gap-2">
                    {[
                      { l: "Shots", v: team.shots },
                      { l: "On Target", v: team.shots_on_target },
                      { l: "xG", v: team.expected_goals },
                      { l: "Passes", v: team.passes },
                      { l: "Pass %", v: team.pass_accuracy != null ? `${team.pass_accuracy}%` : null },
                      { l: "Fouls", v: team.fouls },
                      { l: "Corners", v: team.corners },
                      { l: "Yellow", v: team.yellow_cards },
                      { l: "Red", v: team.red_cards },
                    ].map(({ l, v }) => (
                      <div key={l} className="text-center py-2 px-1 rounded-lg bg-white/2">
                        <div className="text-lg font-display font-600 text-white">{v ?? "—"}</div>
                        <div className="text-xs text-slate-600">{l}</div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* AI Summary */}
        {ai_summary && (
          <>
            <div className="glass-card rounded-xl p-6 mb-6">
              <div className="flex items-center gap-2 mb-4">
                <SectionHeading>AI Tactical Analysis</SectionHeading>
                <PlaceholderBadge />
              </div>
              <p className="text-sm text-slate-300 leading-relaxed mb-3">{ai_summary.tactical_analysis}</p>
              <p className="text-sm text-slate-400 leading-relaxed">{ai_summary.performance_summary}</p>
            </div>
            <div className="grid sm:grid-cols-2 gap-4 mb-6">
              <RecommendationCard title="Team A Strengths" items={ai_summary.team_a_strengths} variant="strength" />
              <RecommendationCard title="Team A Weaknesses" items={ai_summary.team_a_weaknesses} variant="weakness" />
              <RecommendationCard title="Team B Strengths" items={ai_summary.team_b_strengths} variant="strength" />
              <RecommendationCard title="Team B Weaknesses" items={ai_summary.team_b_weaknesses} variant="weakness" />
            </div>
            <div className="grid sm:grid-cols-2 gap-4">
              <RecommendationCard title="Recommendations" items={ai_summary.recommendations} variant="recommendation" />
              <RecommendationCard title="Coaching Suggestions" items={ai_summary.coaching_suggestions} variant="coaching" />
            </div>
          </>
        )}
      </div>
    </div>
  );
}
