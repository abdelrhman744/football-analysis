"use client";

import { CheckCircle2, Circle, Loader2, XCircle, AlertCircle } from "lucide-react";

export type StepStatus = "pending" | "running" | "done" | "error";

export interface AnalysisStep {
  id: string;
  label: string;
  description: string;
  status: StepStatus;
}

interface AnalysisStatusProps {
  steps: AnalysisStep[];
  uploadProgress?: number;
}

/** Steps whose description starts with ⚠ are placeholder — styled differently */
function isPlaceholder(step: AnalysisStep) {
  return step.status === "pending" && step.description.startsWith("⚠");
}

function StepIcon({ step }: { step: AnalysisStep }) {
  if (isPlaceholder(step))
    return <AlertCircle className="h-5 w-5 text-amber-500" />;
  switch (step.status) {
    case "done":    return <CheckCircle2 className="h-5 w-5 text-pitch-500" />;
    case "running": return <Loader2     className="h-5 w-5 text-pitch-400 animate-spin" />;
    case "error":   return <XCircle     className="h-5 w-5 text-red-400" />;
    default:        return <Circle      className="h-5 w-5 text-slate-600" />;
  }
}

function stepLabelColor(step: AnalysisStep): string {
  if (isPlaceholder(step)) return "text-amber-400";
  switch (step.status) {
    case "done":    return "text-white";
    case "running": return "text-pitch-400";
    case "error":   return "text-red-400";
    default:        return "text-slate-500";
  }
}

function stepDescColor(step: AnalysisStep): string {
  if (isPlaceholder(step)) return "text-amber-600";
  return step.status === "pending" ? "text-slate-700" : "text-slate-500";
}

export default function AnalysisStatus({ steps, uploadProgress }: AnalysisStatusProps) {
  return (
    <div className="glass-card rounded-2xl p-6">
      <h3 className="font-display text-lg font-600 text-white mb-6 uppercase tracking-wider">
        Analysis Pipeline
      </h3>

      {/* Upload progress bar */}
      {uploadProgress !== undefined && uploadProgress < 100 && (
        <div className="mb-6">
          <div className="flex justify-between text-xs text-slate-400 mb-2">
            <span>Uploading video...</span>
            <span>{uploadProgress}%</span>
          </div>
          <div className="h-1.5 w-full rounded-full bg-white/5 overflow-hidden">
            <div
              className="h-full rounded-full bg-pitch-500 transition-all duration-300"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
        </div>
      )}

      {/* Steps */}
      <div className="space-y-1">
        {steps.map((step, index) => (
          <div key={step.id}>
            <div
              className={`flex items-start gap-4 p-3 rounded-xl transition-colors ${
                step.status === "running"
                  ? "bg-pitch-500/5 border border-pitch-500/10"
                  : step.status === "done"
                  ? "bg-white/2"
                  : isPlaceholder(step)
                  ? "bg-amber-500/3 border border-amber-500/10"
                  : ""
              }`}
            >
              <div className="mt-0.5 shrink-0">
                <StepIcon step={step} />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between gap-2">
                  <span className={`text-sm font-medium ${stepLabelColor(step)}`}>
                    {step.label}
                  </span>
                  {step.status === "running" && (
                    <span className="text-xs text-pitch-500 font-mono">Processing...</span>
                  )}
                  {step.status === "done" && (
                    <span className="text-xs text-slate-600 font-mono">Complete</span>
                  )}
                  {isPlaceholder(step) && (
                    <span className="text-xs text-amber-600 font-mono">Demo</span>
                  )}
                </div>
                <p className={`text-xs mt-0.5 ${stepDescColor(step)}`}>
                  {/* Strip the ⚠ prefix — it's already shown via the icon */}
                  {step.description.replace(/^⚠\s*/, "")}
                </p>
              </div>
            </div>
            {index < steps.length - 1 && (
              <div
                className={`ml-6 w-px h-3 ${
                  step.status === "done" ? "bg-pitch-500/30" : "bg-white/5"
                }`}
              />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
