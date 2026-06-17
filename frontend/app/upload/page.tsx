"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { CheckCircle2, Film, ArrowRight } from "lucide-react";
import UploadBox from "@/components/UploadBox";
import AnalysisStatus, { AnalysisStep } from "@/components/AnalysisStatus";
import { uploadVideo, startAnalysis, formatBytes, toMediaUrl } from "@/lib/api";

const INITIAL_STEPS: AnalysisStep[] = [
  {
    id: "upload",
    label: "Video Upload",
    description: "Transferring video to server",
    status: "pending",
  },
  {
    id: "detection",
    label: "Player & Ball Detection",
    description: "YOLO model detecting players and ball per frame",
    status: "pending",
  },
  {
    id: "tracking",
    label: "Multi-Object Tracking",
    description: "ByteTrack building movement trajectories",
    status: "pending",
  },
  {
    id: "heatmap",
    label: "Heatmap Generation",
    description: "Computing positional density maps from tracks",
    status: "pending",
  },
  {
    id: "teams",
    label: "Team Classification",
    description: "KMeans jersey-colour classifier assigning players to teams",
    status: "pending",
  },
  {
    id: "stats",
    label: "Match Statistics",
    description: "⚠ Demo only — external statistics API not connected",
    status: "pending",
  },
  {
    id: "ai",
    label: "AI Tactical Analysis",
    description: "⚠ Demo only — LLM tactical model not connected",
    status: "pending",
  },
];

// Steps that are fully real CV steps (can be marked done or error)
const REAL_STEP_IDS       = new Set(["upload", "detection", "tracking", "heatmap", "teams"]);
// Steps that are always placeholder (keep grey, never red)
const PLACEHOLDER_STEP_IDS = new Set(["stats", "ai"]);

type Phase = "idle" | "uploading" | "analyzing" | "complete" | "error";

export default function UploadPage() {
  const router = useRouter();
  const [phase, setPhase]               = useState<Phase>("idle");
  const [steps, setSteps]               = useState<AnalysisStep[]>(INITIAL_STEPS);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [videoId, setVideoId]           = useState<string | null>(null);
  const [errorMsg, setErrorMsg]         = useState<string | null>(null);

  const updateStep = (id: string, status: AnalysisStep["status"], description?: string) => {
    setSteps((prev) =>
      prev.map((s) =>
        s.id === id ? { ...s, status, ...(description ? { description } : {}) } : s
      )
    );
  };

  const handleFileSelected = (file: File) => {
    setSelectedFile(file);
    setPhase("idle");
    setSteps(INITIAL_STEPS);
    setErrorMsg(null);
  };

  const handleStartAnalysis = async () => {
    if (!selectedFile) return;

    setPhase("uploading");
    setErrorMsg(null);
    updateStep("upload", "running");

    try {
      // ── 1. Upload ────────────────────────────────────────────────────────
      const uploadRes = await uploadVideo(selectedFile, (pct) =>
        setUploadProgress(pct)
      );
      const vid = uploadRes.video_id;
      setVideoId(vid);
      localStorage.setItem("lastVideoId", vid);
      localStorage.setItem("lastVideoMeta", JSON.stringify(uploadRes.metadata));
      updateStep("upload", "done");

      // ── 2. Mark real CV steps as running (all happen in one backend pass)
      setPhase("analyzing");
      for (const id of ["detection", "tracking", "heatmap", "teams"]) {
        updateStep(id, "running");
      }

      // ── 3. Call backend — single blocking call ────────────────────────────
      const analysisRes = await startAnalysis(vid);

      if (!analysisRes.success) {
        throw new Error(analysisRes.message || "Analysis reported failure");
      }

      const r = analysisRes.result;

      // ── 4. Mark individual steps based on actual backend result ───────────
      updateStep("detection", r.detection ? "done" : "error");
      updateStep("tracking",  r.tracking  ? "done" : "error");
      updateStep("heatmap",   r.heatmap   ? "done" : "error");

      // Team classification: mark done when backend returned real results
      // (status=completed AND players were actually detected)
      const tc = r.team_classification;
      const teamsReal =
        tc &&
        tc.status === "completed" &&
        (tc.team_a_players.length > 0 || tc.team_b_players.length > 0) &&
        // Exclude the hardcoded placeholder sentinel values
        !(
          tc.team_a_color === "#FFFFFF" &&
          tc.team_b_color === "#003366" &&
          tc.team_a_players.length === 11 &&
          tc.team_b_players.length === 11
        );

      if (teamsReal) {
        updateStep(
          "teams",
          "done",
          `${tc!.team_a_players.length + tc!.team_b_players.length} players classified into 2 teams`,
        );
      } else {
        updateStep(
          "teams",
          tc ? "done" : "error",
          tc
            ? "⚠ Classification ran but used placeholder fallback — check CV model"
            : "⚠ Team classification did not return data",
        );
      }

      // ── 5. Placeholder steps ──────────────────────────────────────────────
      updateStep("stats", "pending", "⚠ Demo placeholder — statistics API not connected");
      updateStep("ai",    "pending", "⚠ Demo placeholder — AI model not connected");

      // ── 6. Persist media URLs for Dashboard ──────────────────────────────
      const videoUrl   = toMediaUrl(analysisRes.processed_video_url);
      const heatmapUrl = toMediaUrl(analysisRes.heatmap_url);

      if (videoUrl)   localStorage.setItem(`videoUrl_${vid}`,   videoUrl);
      if (heatmapUrl) localStorage.setItem(`heatmapUrl_${vid}`, heatmapUrl);

      localStorage.setItem(`analysisResult_${vid}`, JSON.stringify(r));

      setPhase("complete");

    } catch (e: unknown) {
      const msg =
        e instanceof Error ? e.message : "An unexpected error occurred.";
      setErrorMsg(msg);
      setPhase("error");

      setSteps((prev) =>
        prev.map((s) => {
          if (s.status === "running" && REAL_STEP_IDS.has(s.id)) {
            return { ...s, status: "error" };
          }
          if (PLACEHOLDER_STEP_IDS.has(s.id)) {
            return s; // Keep grey — placeholder steps can't "fail"
          }
          return s;
        })
      );
    }
  };

  return (
    <div className="min-h-[calc(100vh-64px)] px-4 py-12">
      <div className="mx-auto max-w-3xl">
        <div className="mb-10">
          <h1 className="font-display text-4xl font-700 text-white mb-2 uppercase tracking-wide">
            Upload & Analyze
          </h1>
          <p className="text-slate-500">
            Upload your football match video to start the AI analysis pipeline.
          </p>
        </div>

        <div className="space-y-6">
          {/* Upload box */}
          <UploadBox
            onFileSelected={handleFileSelected}
            disabled={phase === "uploading" || phase === "analyzing"}
          />

          {/* File info */}
          {selectedFile && phase === "idle" && (
            <div className="flex items-center justify-between glass-card rounded-xl px-5 py-4">
              <div className="flex items-center gap-3">
                <div className="h-10 w-10 rounded-lg bg-pitch-500/10 flex items-center justify-center">
                  <Film className="h-5 w-5 text-pitch-500" />
                </div>
                <div>
                  <p className="text-sm font-medium text-white">{selectedFile.name}</p>
                  <p className="text-xs text-slate-500">{formatBytes(selectedFile.size)}</p>
                </div>
              </div>
              <button
                onClick={handleStartAnalysis}
                className="flex items-center gap-2 rounded-lg bg-pitch-500 px-5 py-2.5 text-sm font-semibold text-navy-950 hover:bg-pitch-400 transition-colors"
              >
                Start Analysis
                <ArrowRight className="h-4 w-4" />
              </button>
            </div>
          )}

          {/* Analysis status */}
          {(phase === "uploading" || phase === "analyzing" || phase === "complete" || phase === "error") && (
            <AnalysisStatus
              steps={steps}
              uploadProgress={phase === "uploading" ? uploadProgress : undefined}
            />
          )}

          {/* Error */}
          {errorMsg && (
            <div className="glass-card rounded-xl px-5 py-4 border border-red-500/20">
              <p className="text-sm text-red-400">{errorMsg}</p>
            </div>
          )}

          {/* Success */}
          {phase === "complete" && (
            <div className="glass-card rounded-xl px-5 py-5 border border-pitch-500/20 bg-pitch-500/3">
              <div className="flex items-center gap-3 mb-4">
                <CheckCircle2 className="h-5 w-5 text-pitch-500" />
                <span className="text-sm font-semibold text-white">Analysis Complete</span>
              </div>
              <p className="text-sm text-slate-400 mb-4">
                All analysis stages have finished. Navigate to the Dashboard to explore results
                or start chatting with the AI analyst.
              </p>
              <div className="flex gap-3">
                <button
                  onClick={() => router.push("/dashboard")}
                  className="flex items-center gap-2 rounded-lg bg-pitch-500 px-5 py-2.5 text-sm font-semibold text-navy-950 hover:bg-pitch-400 transition-colors"
                >
                  View Dashboard
                  <ArrowRight className="h-4 w-4" />
                </button>
                <button
                  onClick={() => router.push("/chat")}
                  className="flex items-center gap-2 rounded-lg border border-white/10 px-5 py-2.5 text-sm font-semibold text-slate-300 hover:border-white/20 hover:text-white transition-colors"
                >
                  Ask AI Analyst
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
