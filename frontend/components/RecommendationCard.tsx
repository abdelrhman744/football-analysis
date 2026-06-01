import { Lightbulb, TrendingUp, TrendingDown, Target } from "lucide-react";

interface RecommendationCardProps {
  title: string;
  items: string[];
  variant?: "recommendation" | "strength" | "weakness" | "coaching";
}

const VARIANT_CONFIG = {
  recommendation: {
    icon: Lightbulb,
    iconBg: "bg-yellow-500/10",
    iconColor: "text-yellow-400",
    dotColor: "bg-yellow-400",
    borderColor: "border-yellow-500/10",
  },
  strength: {
    icon: TrendingUp,
    iconBg: "bg-pitch-500/10",
    iconColor: "text-pitch-500",
    dotColor: "bg-pitch-500",
    borderColor: "border-pitch-500/10",
  },
  weakness: {
    icon: TrendingDown,
    iconBg: "bg-red-500/10",
    iconColor: "text-red-400",
    dotColor: "bg-red-400",
    borderColor: "border-red-500/10",
  },
  coaching: {
    icon: Target,
    iconBg: "bg-blue-500/10",
    iconColor: "text-blue-400",
    dotColor: "bg-blue-400",
    borderColor: "border-blue-500/10",
  },
};

export default function RecommendationCard({
  title,
  items,
  variant = "recommendation",
}: RecommendationCardProps) {
  const config = VARIANT_CONFIG[variant];
  const Icon = config.icon;

  if (items.length === 0) return null;

  return (
    <div className={`glass-card rounded-xl p-5 border ${config.borderColor}`}>
      <div className="flex items-center gap-3 mb-4">
        <div className={`h-8 w-8 rounded-lg flex items-center justify-center ${config.iconBg}`}>
          <Icon className={`h-4 w-4 ${config.iconColor}`} />
        </div>
        <h3 className="font-semibold text-white text-sm">{title}</h3>
      </div>

      <ul className="space-y-2.5">
        {items.map((item, i) => (
          <li key={i} className="flex items-start gap-2.5 text-sm text-slate-400">
            <span className={`mt-1.5 h-1.5 w-1.5 rounded-full shrink-0 ${config.dotColor}`} />
            {item}
          </li>
        ))}
      </ul>
    </div>
  );
}
