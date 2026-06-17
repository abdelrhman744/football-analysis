"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Users, Target, Activity, Zap, Upload,
  Eye, GitBranch, Clock, Video, CheckCircle2, AlertCircle,
  Info, Shield
} from "lucide-react";
import StatsCard from "@/components/StatsCard";
import HeatmapPreview from "@/components/HeatmapPreview";
import RecommendationCard from "@/components/RecommendationCard";
import {
  getAnalysisResult, getCvStatus, FullAnalysisResult,
  TeamClassificationResult,
  CvStatus, formatBytes, BACKEND_BASE
} from "@/lib/api";

// ── Helpers ───────────────────────────────────────────────────────────────

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

function RealBadge() {
  return (
    <span className="ml-2 inline-flex items-center gap-1 text-xs text-pitch-400 border border-pitch-500/20 rounded-full px-2 py-0.5">
      <CheckCircle2 className="h-3 w-3" /> real data
    </span>
  );
}

// ── Video Player ──────────────────────────────────────────────────────────

function VideoPlayer({ videoId }: { videoId: string }) {
  const [loaded, setLoaded] = useState(false);
  const [errored, setErrored] = useState(false);
  const src = `${BACKEND_BASE}/results/${videoId}/processed.mp4`;

  return (
    <div className="glass-card rounded-xl p-6 mb-6">
      <SectionHeading>Processed Video</SectionHeading>
      <div className="aspect-video rounded-lg overflow-hidden bg-black relative">
        <video
          key={src}
          controls
          className="w-full h-full object-contain"
          preload="metadata"
          src={src}
          onLoadedData={() => setLoaded(true)}
          onError={() => setErrored(true)}
        />
        {/* Overlay only shown while video hasn't loaded */}
        {!loaded && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-navy-900/80 pointer-events-none">
            {errored ? (
              <>
                <AlertCircle className="h-12 w-12 text-red-500/60 mb-3" />
                <p className="text-sm text-red-400">Video could not be loaded</p>
                <p className="text-xs text-slate-600 mt-1 font-mono">{src}</p>
              </>
            ) : (
              <>
                <Video className="h-12 w-12 text-slate-700 mb-3" />
                <p className="text-sm text-slate-500">Loading annotated video…</p>
              </>
            )}
          </div>
        )}
      </div>
      <p className="text-xs text-slate-600 mt-2 font-mono">{src}</p>
    </div>
  );
}

// ── Detection Section ─────────────────────────────────────────────────────

function DetectionSection({ detection, usedReal }: {
  detection: FullAnalysisResult["detection"];
  usedReal: boolean;
}) {
  if (!detection) return null;
  return (
    <div className="glass-card rounded-xl p-6 mb-6">
      <div className="flex items-center gap-2 mb-5">
        <SectionHeading>Detection Summary</SectionHeading>
        {usedReal ? <RealBadge /> : <PlaceholderBadge />}
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <StatsCard title="Frames Analyzed" value={detection.total_frames_analyzed.toLocaleString()} icon={Eye} subtitle="CV model output" />
        <StatsCard title="Players Detected" value={detection.players_detected} icon={Users} accent subtitle="Unique player IDs" />
        <StatsCard title="Ball Frames" value={detection.ball_detections.toLocaleString()} icon={Target} subtitle="Frames with ball" />
        <StatsCard title="Status" value={detection.status === "completed" ? "Done" : detection.status} icon={CheckCircle2} accent={detection.status === "completed"} />
      </div>
    </div>
  );
}

// ── Tracking Section ──────────────────────────────────────────────────────

function TrackingSection({ tracking, usedReal }: {
  tracking: FullAnalysisResult["tracking"];
  usedReal: boolean;
}) {
  if (!tracking) return null;
  return (
    <div className="glass-card rounded-xl p-6 mb-6">
      <div className="flex items-center gap-2 mb-5">
        <SectionHeading>Tracking Result</SectionHeading>
        {usedReal ? <RealBadge /> : <PlaceholderBadge />}
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-5">
        <StatsCard title="Players Tracked" value={tracking.total_players_tracked} icon={GitBranch} accent />
        <StatsCard title="Ball Positions" value={tracking.ball_track.length} icon={Activity} subtitle="Recorded positions" />
        <StatsCard title="Status" value={tracking.status === "completed" ? "Done" : tracking.status} icon={CheckCircle2} accent={tracking.status === "completed"} />
        <StatsCard title="Teams" value="2" icon={Users} subtitle="A & B classified" />
      </div>
    </div>
  );
}

// ── Team Classification Section ───────────────────────────────────────────

function TeamClassificationSection({ tc }: { tc: TeamClassificationResult }) {
  const isReal =
    tc.team_a_color !== "#FFFFFF" ||
    tc.team_b_color !== "#003366" ||
    tc.team_a_players.length !== 11 ||
    tc.team_b_players.length !== 11;

  const allPlayers = [
    ...tc.team_a_players.map(p => ({ ...p, team: "A" as const })),
    ...tc.team_b_players.map(p => ({ ...p, team: "B" as const })),
  ].sort((a, b) => a.player_id - b.player_id);

  const avgConfA =
    tc.team_a_players.length > 0
      ? tc.team_a_players.reduce((s, p) => s + p.confidence, 0) / tc.team_a_players.length
      : 0;
  const avgConfB =
    tc.team_b_players.length > 0
      ? tc.team_b_players.reduce((s, p) => s + p.confidence, 0) / tc.team_b_players.length
      : 0;

  return (
    <div className="glass-card rounded-xl p-6 mb-6">
      {/* Header */}
      <div className="flex items-center gap-2 mb-6">
        <SectionHeading>Team Classification</SectionHeading>
        {isReal ? <RealBadge /> : <PlaceholderBadge />}
      </div>

      {/* Diagnostic banner when not real */}
      {!isReal && (
        <div className="mb-5 rounded-lg border border-yellow-500/15 bg-yellow-500/5 px-4 py-3 flex items-start gap-3">
          <AlertCircle className="h-4 w-4 text-yellow-400 shrink-0 mt-0.5" />
          <p className="text-xs text-yellow-300/80">
            Team classification returned placeholder data. This means the CV model
            did not run, or the assignments file was not generated. Re-run analysis
            with model weights present.
          </p>
        </div>
      )}

      {/* Team color cards */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        {[
          {
            label: tc.team_a_label,
            color: tc.team_a_color ?? "#FFFFFF",
            count: tc.team_a_players.length,
            avgConf: avgConfA,
            players: tc.team_a_players,
          },
          {
            label: tc.team_b_label,
            color: tc.team_b_color ?? "#003366",
            count: tc.team_b_players.length,
            avgConf: avgConfB,
            players: tc.team_b_players,
          },
        ].map(({ label, color, count, avgConf, players }) => (
          <div key={label} className="rounded-xl border border-white/5 bg-white/3 p-4">
            <div className="flex items-center gap-3 mb-3">
              <div
                className="h-10 w-10 rounded-lg border border-white/15 shrink-0 shadow-lg"
                style={{ backgroundColor: color }}
                title={color}
              />
              <div>
                <p className="font-semibold text-white">{label}</p>
                <p className="text-xs text-slate-500 font-mono">{color}</p>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2 text-center">
              <div className="rounded-lg bg-white/3 py-2">
                <p className="text-xl font-display font-600 text-white">{count}</p>
                <p className="text-xs text-slate-500">players</p>
              </div>
              <div className="rounded-lg bg-white/3 py-2">
                <p className="text-xl font-display font-600 text-pitch-400">
                  {count > 0 ? `${(avgConf * 100).toFixed(0)}%` : "—"}
                </p>
                <p className="text-xs text-slate-500">avg conf</p>
              </div>
            </div>
            {/* Player ID chips */}
            <div className="mt-3 flex flex-wrap gap-1">
              {players.slice(0, 20).map(p => (
                <span
                  key={p.player_id}
                  className="inline-block rounded px-1.5 py-0.5 text-xs font-mono"
                  style={{
                    backgroundColor: color + "22",
                    color: color,
                    border: `1px solid ${color}44`,
                  }}
                  title={`Confidence: ${(p.confidence * 100).toFixed(1)}%  Color: ${p.dominant_color}`}
                >
                  #{p.player_id}
                </span>
              ))}
              {players.length > 20 && (
                <span className="text-xs text-slate-600 self-center">
                  +{players.length - 20} more
                </span>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Player assignment table */}
      <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">
        Player Assignments
      </h3>
      <div className="overflow-x-auto rounded-lg border border-white/5">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-white/5 bg-white/2">
              <th className="px-4 py-2.5 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Player ID</th>
              <th className="px-4 py-2.5 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Team</th>
              <th className="px-4 py-2.5 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Confidence</th>
              <th className="px-4 py-2.5 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Jersey Color</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/3">
            {allPlayers.map(player => {
              const teamColor = player.team === "A"
                ? (tc.team_a_color ?? "#FFFFFF")
                : (tc.team_b_color ?? "#003366");
              const confPct = Math.round(player.confidence * 100);
              return (
                <tr key={player.player_id} className="hover:bg-white/2 transition-colors">
                  <td className="px-4 py-2.5 font-mono text-slate-300">#{player.player_id}</td>
                  <td className="px-4 py-2.5">
                    <span
                      className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium"
                      style={{
                        backgroundColor: teamColor + "22",
                        color: teamColor,
                        border: `1px solid ${teamColor}44`,
                      }}
                    >
                      <Shield className="h-3 w-3" />
                      {player.team_label}
                    </span>
                  </td>
                  <td className="px-4 py-2.5">
                    <div className="flex items-center gap-2">
                      <div className="flex-1 max-w-[80px] h-1.5 rounded-full bg-white/5">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${confPct}%`,
                            backgroundColor: confPct >= 80 ? "#4ade80" : confPct >= 60 ? "#facc15" : "#f87171",
                          }}
                        />
                      </div>
                      <span className="text-xs text-slate-400 w-10 text-right">{confPct}%</span>
                    </div>
                  </td>
                  <td className="px-4 py-2.5">
                    {player.dominant_color ? (
                      <div className="flex items-center gap-2">
                        <div
                          className="h-4 w-4 rounded border border-white/10 shrink-0"
                          style={{ backgroundColor: player.dominant_color }}
                        />
                        <span className="text-xs font-mono text-slate-500">{player.dominant_color}</span>
                      </div>
                    ) : (
                      <span className="text-xs text-slate-600">—</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Main Dashboard ────────────────────────────────────────────────────────

export default function DashboardPage() {
  const [result, setResult]     = useState<FullAnalysisResult | null>(null);
  const [meta, setMeta]         = useState<Record<string, string> | null>(null);
  const [cvStatus, setCvStatus] = useState<CvStatus | null>(null);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState<string | null>(null);

  useEffect(() => {
    const videoId = localStorage.getItem("lastVideoId");
    const metaRaw = localStorage.getItem("lastVideoMeta");
    if (metaRaw) setMeta(JSON.parse(metaRaw));

    Promise.all([
      videoId
        ? getAnalysisResult(videoId).catch(e => { setError(e.message); return null; })
        : Promise.resolve(null),
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

        {/* CV model warning */}
        {!cvStatus?.cv_model_available && (
          <div className="glass-card rounded-xl px-5 py-4 mb-6 flex items-start gap-3 border border-yellow-500/15">
            <AlertCircle className="h-5 w-5 text-yellow-400 shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-yellow-400 mb-0.5">Placeholder Mode Active</p>
              <p className="text-xs text-slate-500">
                The real CV model weights are not loaded. Results shown are synthetic.
                Place <span className="font-mono">player_ball_detector.pt</span> in{" "}
                <span className="font-mono">backend/models_weights/</span> and restart.
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
        <VideoPlayer videoId={result.video_id} />

        {/* Detection */}
        <DetectionSection detection={detection} usedReal={usedReal} />

        {/* Tracking */}
        <TrackingSection tracking={tracking} usedReal={usedReal} />

        {/* Heatmap */}
        <div className="mb-6">
          <HeatmapPreview matrix={heatmap?.heatmap_matrix ?? null} />
          {heatmap?.team_a_heatmap_url && (
            <div className="mt-3 glass-card rounded-xl p-4">
              <p className="text-xs text-slate-500 mb-2">Real heatmap images from CV model:</p>
              <div className="grid grid-cols-2 gap-3">
                {heatmap.team_a_heatmap_url && (
                  <div>
                    <p className="text-xs text-slate-600 mb-1">Players</p>
                    <img src={`${BACKEND_BASE}${heatmap.team_a_heatmap_url}`} alt="Player heatmap" className="w-full rounded-lg object-cover" />
                  </div>
                )}
                {heatmap.ball_heatmap_url && (
                  <div>
                    <p className="text-xs text-slate-600 mb-1">Ball</p>
                    <img src={`${BACKEND_BASE}${heatmap.ball_heatmap_url}`} alt="Ball heatmap" className="w-full rounded-lg object-cover" />
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Team Classification — real component */}
        {team_classification ? (
          <TeamClassificationSection tc={team_classification} />
        ) : (
          <div className="glass-card rounded-xl p-6 mb-6 flex items-center gap-3">
            <AlertCircle className="h-5 w-5 text-slate-600 shrink-0" />
            <p className="text-sm text-slate-500">Team classification data unavailable.</p>
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
