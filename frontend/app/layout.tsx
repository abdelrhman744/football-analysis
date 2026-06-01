import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";

export const metadata: Metadata = {
  title: "MatchVision — Football Match Analysis",
  description: "AI-powered football match analysis platform. Upload your match video and get tactical insights, player tracking, heatmaps, and performance analytics.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-navy-950 text-slate-200 font-body">
        <Navbar />
        <main>{children}</main>
      </body>
    </html>
  );
}
