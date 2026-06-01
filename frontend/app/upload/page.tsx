"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { CheckCircle2, Film, ArrowRight } from "lucide-react";
import UploadBox from "@/components/UploadBox";
import AnalysisStatus, { AnalysisStep } from "@/components/AnalysisStatus";
import { uploadVideo, startAnalysis, formatBytes, toMediaUrl } from "@/lib/api";

// "placeholder" re-uses the "pending" visual but with a distinct label.
// We extend StepStatus locally — AnalysisStatus.tsx already renders
// "pending" as a grey circle, which is perfect for placeholder steps.
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
    description: "Demo only — kit colour classifier not connected yet",
    status: "pending",
  },
  {
    id: "stats",
    label: "Match Statistics",
    description: "Demo only — external statistics API not connected",
    status: "pending",
  },
  {
    id: "ai",
    label: "AI Tactical Analysis",
    description: "Demo only — LLM tactical model not connected",
    status: "pending",
  },
];

// Which steps are driven by real CV output vs placeholder services
const REAL_STEP_IDS  = new Set(["upload", "detection", "tracking", "heatmap"]);
const PLACEHOLDER_STEP_IDS = new Set(["teams", "stats", "ai"]);

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
      for (const id of ["detection", "tracking", "heatmap"]) {
        updateStep(id, "running");
      }

      // ── 3. Call backend — single blocking call, real work happens here ──
      const analysisRes = await startAnalysis(vid);

      // ── 4. Handle success=false (backend ran but reported failure) ───────
      if (!analysisRes.success) {
        throw new Error(analysisRes.message || "Analysis reported failure");
      }

      // ── 5. Mark real CV steps done ────────────────────────────────────────
      // Check each sub-result; if it came back, mark done, else error.
      const r = analysisRes.result;

      updateStep("detection", r.detection ? "done" : "error");
      updateStep("tracking",  r.tracking  ? "done" : "error");
      updateStep("heatmap",   r.heatmap   ? "done" : "error");

      // ── 6. Placeholder steps — always show as "pending" (grey circle) ────
      // Update description to make placeholder status obvious in the UI.
      updateStep("teams", "pending", "⚠ Demo placeholder — team classifier not connected");
      updateStep("stats", "pending", "⚠ Demo placeholder — statistics API not connected");
      updateStep("ai",    "pending", "⚠ Demo placeholder — AI model not connected");

      // ── 7. Persist media URLs for Dashboard ──────────────────────────────
      const videoUrl   = toMediaUrl(analysisRes.processed_video_url);
      const heatmapUrl = toMediaUrl(analysisRes.heatmap_url);

      if (videoUrl)   localStorage.setItem(`videoUrl_${vid}`,   videoUrl);
      if (heatmapUrl) localStorage.setItem(`heatmapUrl_${vid}`, heatmapUrl);

      // Also persist the full result for the dashboard
      localStorage.setItem(`analysisResult_${vid}`, JSON.stringify(r));

      setPhase("complete");

    } catch (e: unknown) {
      const msg =
        e instanceof Error ? e.message : "An unexpected error occurred.";
      setErrorMsg(msg);
      setPhase("error");

      // Only fail steps that were genuinely running (real CV steps).
      // Never turn placeholder steps red — they were never running.
      setSteps((prev) =>
        prev.map((s) => {
          if (s.status === "running" && REAL_STEP_IDS.has(s.id)) {
            return { ...s, status: "error" };
          }
          if (PLACEHOLDER_STEP_IDS.has(s.id)) {
            // Keep grey — placeholder steps can't "fail"
            return s;
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
