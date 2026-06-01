import { LucideIcon } from "lucide-react";

interface StatsCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: LucideIcon;
  trend?: "up" | "down" | "neutral";
  trendLabel?: string;
  accent?: boolean;
}

export default function StatsCard({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  trendLabel,
  accent = false,
}: StatsCardProps) {
  const trendColor =
    trend === "up" ? "text-pitch-400" : trend === "down" ? "text-red-400" : "text-slate-500";

  return (
    <div
      className={`glass-card rounded-xl p-5 transition-all hover:border-white/10 ${
        accent ? "border-pitch-500/20 bg-pitch-500/3" : ""
      }`}
    >
      <div className="flex items-start justify-between mb-3">
        <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">{title}</span>
        <div
          className={`h-8 w-8 rounded-lg flex items-center justify-center ${
            accent ? "bg-pitch-500/15" : "bg-white/5"
          }`}
        >
          <Icon className={`h-4 w-4 ${accent ? "text-pitch-500" : "text-slate-400"}`} />
        </div>
      </div>

      <div className="text-2xl font-display font-700 text-white mb-1">
        {value}
      </div>

      {subtitle && (
        <p className="text-xs text-slate-500">{subtitle}</p>
      )}

      {trendLabel && (
        <p className={`text-xs mt-2 font-medium ${trendColor}`}>{trendLabel}</p>
      )}
    </div>
  );
}
