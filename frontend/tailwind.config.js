/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        navy: {
          950: "#040d1a",
          900: "#071428",
          800: "#0a1e3d",
          700: "#0d2850",
          600: "#103263",
        },
        pitch: {
          500: "#00c853",
          400: "#00e676",
          300: "#69f0ae",
        },
      },
      fontFamily: {
        display: ["'Barlow Condensed'", "sans-serif"],
        body: ["'DM Sans'", "sans-serif"],
        mono: ["'JetBrains Mono'", "monospace"],
      },
      backgroundImage: {
        "pitch-gradient": "linear-gradient(135deg, #040d1a 0%, #071428 50%, #0a1e3d 100%)",
        "card-gradient": "linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.06) 100%)",
        "green-glow": "radial-gradient(ellipse at center, rgba(0,200,83,0.15) 0%, transparent 70%)",
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "fade-up": "fadeUp 0.5s ease-out forwards",
        "progress": "progress 1.5s ease-in-out infinite",
      },
      keyframes: {
        fadeUp: {
          "0%": { opacity: "0", transform: "translateY(16px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        progress: {
          "0%": { width: "0%" },
          "50%": { width: "70%" },
          "100%": { width: "100%" },
        },
      },
    },
  },
  plugins: [],
};
