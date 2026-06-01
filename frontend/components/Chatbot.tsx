"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Loader2, Database } from "lucide-react";
import { sendChatMessage, ChatMessage } from "@/lib/api";

interface ChatbotProps {
  videoId: string | null;
}

const SUGGESTED_QUESTIONS = [
  "What was the possession breakdown?",
  "How did the pressing patterns differ?",
  "Which team controlled the final third?",
  "What defensive weaknesses were identified?",
  "Describe the attacking patterns of Team A.",
];

export default function Chatbot({ videoId }: ChatbotProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content:
        "Hello. I am the MatchVision AI analyst. Once a video has been uploaded and analyzed, you can ask me detailed questions about the match — tactics, player movement, heatmaps, or statistical comparisons. How can I help?",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async (question?: string) => {
    const text = question ?? input.trim();
    if (!text || loading) return;

    if (!videoId) {
      setError("No video selected. Please upload and analyze a match first.");
      return;
    }

    setError(null);
    const userMsg: ChatMessage = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await sendChatMessage(videoId, text, messages);
      setMessages((prev) => [...prev, { role: "assistant", content: res.answer }]);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Failed to get response.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full glass-card rounded-2xl overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-3 px-5 py-4 border-b border-white/5">
        <div className="h-8 w-8 rounded-lg bg-pitch-500/10 flex items-center justify-center">
          <Bot className="h-4 w-4 text-pitch-500" />
        </div>
        <div>
          <h3 className="text-sm font-semibold text-white">AI Match Analyst</h3>
          <p className="text-xs text-slate-500">
            {videoId ? `Video: ${videoId.slice(0, 8)}...` : "No video selected"}
          </p>
        </div>
        {videoId && (
          <div className="ml-auto flex items-center gap-1.5 text-xs text-pitch-500">
            <Database className="h-3 w-3" />
            <span>Match data loaded</span>
          </div>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex gap-3 ${msg.role === "user" ? "flex-row-reverse" : ""}`}
          >
            <div
              className={`shrink-0 h-8 w-8 rounded-full flex items-center justify-center ${
                msg.role === "assistant" ? "bg-pitch-500/10" : "bg-white/5"
              }`}
            >
              {msg.role === "assistant" ? (
                <Bot className="h-4 w-4 text-pitch-500" />
              ) : (
                <User className="h-4 w-4 text-slate-400" />
              )}
            </div>
            <div
              className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                msg.role === "assistant"
                  ? "bg-white/4 text-slate-200 rounded-tl-sm"
                  : "bg-pitch-500/10 text-white border border-pitch-500/15 rounded-tr-sm"
              }`}
            >
              {msg.content}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex gap-3">
            <div className="shrink-0 h-8 w-8 rounded-full bg-pitch-500/10 flex items-center justify-center">
              <Bot className="h-4 w-4 text-pitch-500" />
            </div>
            <div className="bg-white/4 rounded-2xl rounded-tl-sm px-4 py-3 flex items-center gap-2">
              <Loader2 className="h-3 w-3 text-pitch-500 animate-spin" />
              <span className="text-xs text-slate-500">Analyzing match data...</span>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Suggested questions */}
      {messages.length <= 1 && !loading && (
        <div className="px-4 pb-3">
          <p className="text-xs text-slate-600 mb-2">Suggested questions:</p>
          <div className="flex flex-wrap gap-2">
            {SUGGESTED_QUESTIONS.map((q) => (
              <button
                key={q}
                onClick={() => handleSend(q)}
                className="text-xs text-slate-400 border border-white/8 rounded-full px-3 py-1 hover:border-pitch-500/30 hover:text-pitch-400 transition-colors"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mx-4 mb-2 text-xs text-red-400 bg-red-500/5 border border-red-500/15 rounded-lg px-3 py-2">
          {error}
        </div>
      )}

      {/* Input */}
      <div className="px-4 pb-4">
        <div className="flex items-end gap-3 rounded-xl border border-white/8 bg-white/3 px-4 py-3 focus-within:border-pitch-500/30 transition-colors">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about tactics, heatmaps, player performance..."
            rows={1}
            className="flex-1 bg-transparent text-sm text-white placeholder-slate-600 resize-none outline-none leading-relaxed"
            style={{ maxHeight: "120px" }}
          />
          <button
            onClick={() => handleSend()}
            disabled={!input.trim() || loading}
            className="shrink-0 h-8 w-8 rounded-lg bg-pitch-500 flex items-center justify-center hover:bg-pitch-400 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <Send className="h-3.5 w-3.5 text-navy-950" />
          </button>
        </div>
        <p className="text-xs text-slate-700 mt-1.5 text-center">Press Enter to send</p>
      </div>
    </div>
  );
}
