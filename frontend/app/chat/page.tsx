"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Upload, AlertCircle } from "lucide-react";
import Chatbot from "@/components/Chatbot";

export default function ChatPage() {
  const [videoId, setVideoId] = useState<string | null>(null);
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    setVideoId(localStorage.getItem("lastVideoId"));
    setChecked(true);
  }, []);

  if (!checked) return null;

  return (
    <div className="min-h-[calc(100vh-64px)] px-4 py-12">
      <div className="mx-auto max-w-4xl">
        <div className="mb-8">
          <h1 className="font-display text-4xl font-700 text-white uppercase tracking-wide mb-2">
            AI Match Analyst
          </h1>
          <p className="text-slate-500">
            Ask natural language questions about your match. The AI synthesizes tracking data,
            heatmaps, team classification, and statistics.
          </p>
        </div>

        {!videoId && (
          <div className="glass-card rounded-xl p-5 mb-6 flex items-start gap-3 border border-yellow-500/15">
            <AlertCircle className="h-5 w-5 text-yellow-400 shrink-0 mt-0.5" />
            <div>
              <p className="text-sm text-yellow-400 font-medium mb-1">No match analyzed yet</p>
              <p className="text-sm text-slate-500 mb-3">
                Upload and analyze a match video first to get context-aware answers.
              </p>
              <Link
                href="/upload"
                className="inline-flex items-center gap-2 text-xs text-pitch-400 border border-pitch-500/20 rounded-lg px-3 py-1.5 hover:bg-pitch-500/5 transition-colors"
              >
                <Upload className="h-3 w-3" />
                Upload a match
              </Link>
            </div>
          </div>
        )}

        <div style={{ height: "calc(100vh - 320px)", minHeight: "500px" }}>
          <Chatbot videoId={videoId} />
        </div>
      </div>
    </div>
  );
}
