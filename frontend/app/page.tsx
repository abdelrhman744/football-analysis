import Link from "next/link";
import { ArrowRight, Eye, GitBranch, BarChart2, MessageSquare, Cpu, Users } from "lucide-react";

const FEATURES = [
  {
    icon: Eye,
    title: "Computer Vision Detection",
    description:
      "AI-powered player and ball detection across all frames. Precision bounding boxes with confidence scoring for every object on the pitch.",
  },
  {
    icon: GitBranch,
    title: "Movement Tracking",
    description:
      "Multi-object tracking assigns unique IDs to players across frames, computing distance covered, speed profiles, and spatial trajectories.",
  },
  {
    icon: BarChart2,
    title: "Heatmap Generation",
    description:
      "Kernel density estimation maps reveal positional tendencies and territorial dominance for each team and individual player.",
  },
  {
    icon: Users,
    title: "Team Classification",
    description:
      "Kit color analysis automatically separates players into teams, enabling team-level spatial analytics without manual labeling.",
  },
  {
    icon: Cpu,
    title: "Live Match Statistics",
    description:
      "Real-time integration with football data APIs brings possession, xG, pass networks, and event data into a unified dashboard.",
  },
  {
    icon: MessageSquare,
    title: "AI Tactical Chatbot",
    description:
      "Ask natural language questions about any aspect of the match. The AI synthesizes all data sources into actionable coaching insights.",
  },
];

const PIPELINE_STEPS = [
  { step: "01", label: "Upload Video", desc: "Match footage ingested securely" },
  { step: "02", label: "CV Detection", desc: "Players & ball located per frame" },
  { step: "03", label: "Tracking", desc: "Movement trajectories built" },
  { step: "04", label: "Classification", desc: "Teams identified by kit" },
  { step: "05", label: "Stats Fetch", desc: "External API data merged" },
  { step: "06", label: "AI Analysis", desc: "Tactical report generated" },
];

export default function HomePage() {
  return (
    <div className="relative overflow-hidden">
      {/* Grid background */}
      <div className="absolute inset-0 pitch-bg pointer-events-none" />

      {/* Hero */}
      <section className="relative min-h-[calc(100vh-64px)] flex flex-col items-center justify-center px-4 text-center">
        <div className="absolute inset-0 bg-green-glow pointer-events-none" />

        <div className="inline-flex items-center gap-2 rounded-full border border-pitch-500/20 bg-pitch-500/5 px-4 py-1.5 text-xs font-medium text-pitch-400 mb-8 tracking-wider uppercase">
          <span className="h-1.5 w-1.5 rounded-full bg-pitch-500 animate-pulse" />
          AI-Powered Football Analytics
        </div>

        <h1 className="font-display text-5xl sm:text-7xl lg:text-8xl font-800 tracking-tight text-white mb-6 leading-none">
          UNDERSTAND
          <br />
          <span className="text-pitch-500">EVERY MOMENT</span>
          <br />
          OF THE MATCH
        </h1>

        <p className="max-w-2xl text-lg text-slate-400 mb-10 leading-relaxed">
          Upload your football match footage and receive comprehensive AI analysis —
          player tracking, heatmaps, team classification, and tactical insights powered
          by computer vision and large language models.
        </p>

        <div className="flex flex-col sm:flex-row gap-4 items-center">
          <Link
            href="/upload"
            className="flex items-center gap-2 rounded-lg bg-pitch-500 px-8 py-3.5 text-sm font-semibold text-navy-950 hover:bg-pitch-400 transition-colors glow-green"
          >
            Analyze a Match
            <ArrowRight className="h-4 w-4" />
          </Link>
          <Link
            href="/dashboard"
            className="flex items-center gap-2 rounded-lg border border-white/10 px-8 py-3.5 text-sm font-semibold text-slate-300 hover:border-white/20 hover:text-white transition-colors"
          >
            View Dashboard
          </Link>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-8 flex flex-col items-center gap-2 text-slate-600 text-xs">
          <span>Scroll to explore</span>
          <div className="w-px h-8 bg-gradient-to-b from-slate-600 to-transparent" />
        </div>
      </section>

      {/* Pipeline steps */}
      <section className="relative py-20 px-4">
        <div className="mx-auto max-w-7xl">
          <h2 className="font-display text-3xl font-700 text-center text-white mb-4 uppercase tracking-wider">
            The Analysis Pipeline
          </h2>
          <p className="text-center text-slate-500 mb-12 max-w-xl mx-auto">
            Six stages of automated intelligence, from raw video to tactical report.
          </p>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
            {PIPELINE_STEPS.map(({ step, label, desc }, i) => (
              <div key={step} className="relative flex flex-col items-center text-center p-4">
                {/* Connector line */}
                {i < PIPELINE_STEPS.length - 1 && (
                  <div className="hidden lg:block absolute top-6 left-[calc(50%+24px)] right-0 h-px bg-gradient-to-r from-pitch-500/30 to-transparent" />
                )}
                <div className="h-12 w-12 rounded-full border border-pitch-500/30 bg-pitch-500/5 flex items-center justify-center text-pitch-500 font-display font-700 text-sm mb-3">
                  {step}
                </div>
                <span className="text-sm font-semibold text-white mb-1">{label}</span>
                <span className="text-xs text-slate-500">{desc}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features grid */}
      <section className="relative py-20 px-4 border-t border-white/5">
        <div className="mx-auto max-w-7xl">
          <h2 className="font-display text-3xl font-700 text-center text-white mb-4 uppercase tracking-wider">
            Capabilities
          </h2>
          <p className="text-center text-slate-500 mb-16 max-w-xl mx-auto">
            A complete platform for football performance analysis, built for coaches,
            analysts, and data scientists.
          </p>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {FEATURES.map(({ icon: Icon, title, description }) => (
              <div
                key={title}
                className="glass-card rounded-xl p-6 hover:border-white/10 transition-colors group"
              >
                <div className="mb-4 h-10 w-10 rounded-lg bg-pitch-500/10 flex items-center justify-center group-hover:bg-pitch-500/15 transition-colors">
                  <Icon className="h-5 w-5 text-pitch-500" />
                </div>
                <h3 className="font-semibold text-white mb-2">{title}</h3>
                <p className="text-sm text-slate-400 leading-relaxed">{description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="relative py-24 px-4 border-t border-white/5">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="font-display text-4xl font-700 text-white mb-4 uppercase">
            Ready to Analyze?
          </h2>
          <p className="text-slate-400 mb-8">
            Upload your first match video and let the AI do the work.
          </p>
          <Link
            href="/upload"
            className="inline-flex items-center gap-2 rounded-lg bg-pitch-500 px-10 py-4 text-sm font-semibold text-navy-950 hover:bg-pitch-400 transition-colors glow-green"
          >
            Upload Match Video
            <ArrowRight className="h-4 w-4" />
          </Link>
        </div>
      </section>
    </div>
  );
}
