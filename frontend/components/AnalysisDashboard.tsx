"use client";

import { FullAnalysisResult, toMediaUrl } from "@/lib/api";

interface Props {
  result: FullAnalysisResult;
  /** Full URL — pass toMediaUrl(analysisRes.processed_video_url) */
  videoUrl?: string | null;
  /** Full URL — pass toMediaUrl(analysisRes.heatmap_url) */
  heatmapUrl?: string | null;
}

// ── Badges ────────────────────────────────────────────────────────────────────

function LiveBadge() {
  return (
    <span className="inline-flex items-center gap-1 rounded bg-green-100 px-2 py-0.5 text-xs font-semibold text-green-800 border border-green-300">
      ✓ Live CV Output
    </span>
  );
}

function DemoBadge() {
  return (
    <span className="inline-flex items-center gap-1 rounded bg-amber-100 px-2 py-0.5 text-xs font-semibold text-amber-800 border border-amber-300">
      ⚠ Demo / Not Connected
    </span>
  );
}

// ── Shared primitives ─────────────────────────────────────────────────────────

function SectionHeader({ title, live, note }: { title: string; live: boolean; note?: string }) {
  return (
    <div className="flex flex-wrap items-center gap-3 mb-3">
      <h2 className="text-lg font-semibold text-slate-800">{title}</h2>
      {live ? <LiveBadge /> : <DemoBadge />}
      {!live && note && <p className="text-xs text-slate-500 italic">{note}</p>}
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string | number | null | undefined }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <p className="text-xs text-slate-500 mb-1">{label}</p>
      {value !== null && value !== undefined ? (
        <p className="text-2xl font-bold text-slate-800">{value}</p>
      ) : (
        <p className="text-sm text-slate-400 italic">—</p>
      )}
    </div>
  );
}

function PlaceholderSection({ title, note }: { title: string; note: string }) {
  return (
    <section className="rounded-lg border border-amber-200 bg-amber-50 p-5">
      <SectionHeader title={title} live={false} />
      <p className="text-sm text-amber-800">{note}</p>
    </section>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function AnalysisDashboard({ result, videoUrl, heatmapUrl }: Props) {
  const det      = result.detection;
  const tracking = result.tracking;
  const heatmap  = result.heatmap;

  // Resolve heatmap image URL: prefer explicit prop, fall back to schema field
  const resolvedHeatmap =
    heatmapUrl ?? toMediaUrl(heatmap?.team_a_heatmap_url) ?? null;

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-8">

      {/* ── CV Detection Stats (real) ────────────────────────────────────── */}
      <section>
        <SectionHeader title="Player & Ball Detection" live={true} />
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard label="Frames Analyzed"  value={det?.total_frames_analyzed} />
          <StatCard label="Total Detections" value={det?.ball_detections} />
          <StatCard label="Players Tracked"  value={det?.players_detected} />
          <StatCard
            label="Tracking IDs"
            value={tracking?.total_players_tracked}
          />
        </div>
      </section>

      {/* ── Processed Video (real) ───────────────────────────────────────── */}
      {videoUrl ? (
        <section>
          <SectionHeader title="Annotated Output Video" live={true} />
          <div className="rounded-lg overflow-hidden border border-slate-200 bg-black">
            <video controls className="w-full max-h-[480px]" src={videoUrl}>
              Your browser does not support the video tag.
            </video>
          </div>
        </section>
      ) : (
        <section className="rounded-lg border border-slate-200 bg-slate-50 p-5">
          <SectionHeader title="Annotated Output Video" live={true} />
          <p className="text-sm text-slate-500">
            Processed video not available — check that{" "}
            <code className="text-xs bg-slate-100 px-1 rounded">
              results/&#123;video_id&#125;/processed.mp4
            </code>{" "}
            exists and the backend is serving static files.
          </p>
        </section>
      )}

      {/* ── Heatmap (real) ──────────────────────────────────────────────── */}
      {resolvedHeatmap ? (
        <section>
          <SectionHeader title="Player Movement Heatmap" live={true} />
          <div className="rounded-lg overflow-hidden border border-slate-200">
            <img
              src={resolvedHeatmap}
              alt="Player movement heatmap"
              className="w-full object-contain"
            />
          </div>
        </section>
      ) : (
        <section className="rounded-lg border border-slate-200 bg-slate-50 p-5">
          <SectionHeader title="Player Movement Heatmap" live={true} />
          <p className="text-sm text-slate-500">
            Heatmap not generated — video may be too short or tracking produced no data.
          </p>
        </section>
      )}

      {/* ── Placeholder sections ─────────────────────────────────────────── */}
      <PlaceholderSection
        title="Team Classification"
        note="Kit colour clustering is not yet implemented. Players are not assigned to real teams."
      />
      <PlaceholderSection
        title="Ball Possession"
        note="Possession tracking requires ball detection combined with zone attribution logic."
      />
      <PlaceholderSection
        title="Match Statistics"
        note="Shots, passes, fouls, and xG would require an external statistics API or additional CV models."
      />
      <PlaceholderSection
        title="AI Tactical Analysis"
        note="Formation recognition, strengths/weaknesses, and recommendations require a connected LLM with match context."
      />

    </div>
  );
}
