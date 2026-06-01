"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, BarChart2, MessageSquare, Upload } from "lucide-react";

const NAV_LINKS = [
  { href: "/", label: "Home", icon: Activity },
  { href: "/upload", label: "Analyze", icon: Upload },
  { href: "/dashboard", label: "Dashboard", icon: BarChart2 },
  { href: "/chat", label: "AI Chat", icon: MessageSquare },
];

export default function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="sticky top-0 z-50 border-b border-white/5 bg-navy-950/80 backdrop-blur-xl">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-3 group">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-pitch-500/10 border border-pitch-500/20 group-hover:bg-pitch-500/20 transition-colors">
              <Activity className="h-4 w-4 text-pitch-500" />
            </div>
            <span className="font-display text-xl font-700 tracking-wide text-white">
              MATCH<span className="text-pitch-500">VISION</span>
            </span>
          </Link>

          {/* Nav links */}
          <div className="flex items-center gap-1">
            {NAV_LINKS.map(({ href, label, icon: Icon }) => {
              const active = pathname === href;
              return (
                <Link
                  key={href}
                  href={href}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                    active
                      ? "bg-pitch-500/10 text-pitch-400 border border-pitch-500/20"
                      : "text-slate-400 hover:text-white hover:bg-white/5"
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span className="hidden sm:block">{label}</span>
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
}
