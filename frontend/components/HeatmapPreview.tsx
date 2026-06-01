"use client";

interface HeatmapPreviewProps {
  matrix: number[][] | null;
  title?: string;
}

function getHeatColor(value: number): string {
  // 0 = cool blue, 0.5 = yellow, 1.0 = hot red-green
  if (value < 0.25) {
    const t = value / 0.25;
    return `rgba(0, ${Math.round(50 + t * 80)}, ${Math.round(150 + t * 50)}, ${0.3 + t * 0.3})`;
  } else if (value < 0.5) {
    const t = (value - 0.25) / 0.25;
    return `rgba(${Math.round(t * 200)}, ${Math.round(130 + t * 70)}, ${Math.round(100 - t * 80)}, ${0.5 + t * 0.2})`;
  } else if (value < 0.75) {
    const t = (value - 0.5) / 0.25;
    return `rgba(${Math.round(200 + t * 55)}, ${Math.round(200 - t * 100)}, ${Math.round(20 - t * 10)}, ${0.65 + t * 0.2})`;
  } else {
    const t = (value - 0.75) / 0.25;
    return `rgba(${Math.round(255)}, ${Math.round(100 - t * 80)}, ${Math.round(10 - t * 5)}, ${0.85})`;
  }
}

export default function HeatmapPreview({ matrix, title = "Positional Heatmap" }: HeatmapPreviewProps) {
  if (!matrix || matrix.length === 0) {
    return (
      <div className="glass-card rounded-xl p-6">
        <h3 className="font-display text-sm font-600 text-slate-400 uppercase tracking-wider mb-4">
          {title}
        </h3>
        <div className="aspect-[16/9] rounded-lg bg-white/2 flex items-center justify-center">
          <span className="text-sm text-slate-600">No heatmap data available</span>
        </div>
      </div>
    );
  }

  const rows = matrix.length;
  const cols = matrix[0].length;

  return (
    <div className="glass-card rounded-xl p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-display text-sm font-600 text-white uppercase tracking-wider">
          {title}
        </h3>
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span>Low</span>
          <div className="w-20 h-2 rounded-full" style={{
            background: "linear-gradient(to right, rgba(0,80,150,0.5), rgba(200,200,0,0.7), rgba(255,40,10,0.9))"
          }} />
          <span>High</span>
        </div>
      </div>

      {/* Pitch outline */}
      <div className="relative aspect-[16/9] rounded-lg overflow-hidden bg-[#1a3a1a] border border-white/5">
        {/* Pitch markings */}
        <div className="absolute inset-0">
          {/* Centre line */}
          <div className="absolute top-0 bottom-0 left-1/2 w-px bg-white/10" />
          {/* Centre circle */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 h-[30%] w-[15%] rounded-full border border-white/10" />
          {/* Penalty boxes */}
          <div className="absolute top-[20%] bottom-[20%] left-0 w-[14%] border-r border-white/10" />
          <div className="absolute top-[20%] bottom-[20%] right-0 w-[14%] border-l border-white/10" />
          {/* Goals */}
          <div className="absolute top-[38%] bottom-[38%] left-0 w-[3%] border-r border-white/15" />
          <div className="absolute top-[38%] bottom-[38%] right-0 w-[3%] border-l border-white/15" />
        </div>

        {/* Heatmap grid overlay */}
        <div
          className="absolute inset-0 grid"
          style={{ gridTemplateRows: `repeat(${rows}, 1fr)`, gridTemplateColumns: `repeat(${cols}, 1fr)` }}
        >
          {matrix.map((row, ri) =>
            row.map((val, ci) => (
              <div
                key={`${ri}-${ci}`}
                style={{ backgroundColor: getHeatColor(val) }}
                title={`Zone [${ri},${ci}]: ${(val * 100).toFixed(0)}%`}
              />
            ))
          )}
        </div>
      </div>

      <p className="text-xs text-slate-600 mt-2 text-center">
        {rows} x {cols} zone grid — combined team activity
      </p>
    </div>
  );
}
