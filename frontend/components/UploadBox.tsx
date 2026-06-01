"use client";

import { useState, useRef, DragEvent, ChangeEvent } from "react";
import { Upload, Film, AlertCircle } from "lucide-react";

interface UploadBoxProps {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
}

const ACCEPTED_TYPES = ["video/mp4", "video/quicktime", "video/x-msvideo", "video/x-matroska", "video/webm"];
const MAX_SIZE_MB = 500;

export default function UploadBox({ onFileSelected, disabled = false }: UploadBoxProps) {
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const validate = (file: File): string | null => {
    if (!ACCEPTED_TYPES.includes(file.type) && !file.name.match(/\.(mp4|mov|avi|mkv|webm)$/i)) {
      return "Unsupported file format. Please upload MP4, MOV, AVI, MKV, or WebM.";
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      return `File too large. Maximum size is ${MAX_SIZE_MB} MB.`;
    }
    return null;
  };

  const handleFile = (file: File) => {
    const err = validate(file);
    if (err) {
      setError(err);
      return;
    }
    setError(null);
    onFileSelected(file);
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    if (disabled) return;
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const onDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (!disabled) setDragging(true);
  };

  const onDragLeave = () => setDragging(false);

  const onChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  return (
    <div className="w-full">
      <div
        onClick={() => !disabled && inputRef.current?.click()}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        className={`relative flex flex-col items-center justify-center rounded-2xl border-2 border-dashed p-12 text-center transition-all cursor-pointer ${
          disabled
            ? "cursor-not-allowed opacity-50 border-white/5 bg-white/2"
            : dragging
            ? "border-pitch-500 bg-pitch-500/5 scale-[1.01]"
            : "border-white/10 bg-white/2 hover:border-pitch-500/50 hover:bg-white/3"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          accept=".mp4,.mov,.avi,.mkv,.webm"
          onChange={onChange}
          disabled={disabled}
        />

        <div className={`mb-4 h-16 w-16 rounded-2xl flex items-center justify-center transition-colors ${
          dragging ? "bg-pitch-500/20" : "bg-white/5"
        }`}>
          <Film className={`h-8 w-8 transition-colors ${dragging ? "text-pitch-500" : "text-slate-500"}`} />
        </div>

        <h3 className="text-lg font-semibold text-white mb-2">
          {dragging ? "Release to upload" : "Drop your match video here"}
        </h3>
        <p className="text-sm text-slate-500 mb-4">
          or click to browse files
        </p>

        <div className="flex items-center gap-2 text-xs text-slate-600">
          <Upload className="h-3 w-3" />
          <span>MP4, MOV, AVI, MKV, WebM — up to {MAX_SIZE_MB} MB</span>
        </div>
      </div>

      {error && (
        <div className="mt-3 flex items-center gap-2 rounded-lg border border-red-500/20 bg-red-500/5 px-4 py-3 text-sm text-red-400">
          <AlertCircle className="h-4 w-4 shrink-0" />
          {error}
        </div>
      )}
    </div>
  );
}
