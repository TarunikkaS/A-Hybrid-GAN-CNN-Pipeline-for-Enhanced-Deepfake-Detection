import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      // Terminal CLI Color Palette
      colors: {
        terminal: {
          bg: "#0a0a0a",
          primary: "#33ff00",
          secondary: "#ffb000",
          muted: "#1f521f",
          accent: "#33ff00",
          error: "#ff3333",
          border: "#1f521f",
          dimmed: "#0d1f0d",
        },
      },
      // Monospace Typography
      fontFamily: {
        mono: [
          "JetBrains Mono",
          "Fira Code",
          "VT323",
          "Consolas",
          "Monaco",
          "monospace",
        ],
      },
      // No border radius - brutalist design
      borderRadius: {
        none: "0px",
      },
      // Text glow effect for phosphor persistence
      textShadow: {
        glow: "0 0 5px rgba(51, 255, 0, 0.5)",
        "glow-amber": "0 0 5px rgba(255, 176, 0, 0.5)",
        "glow-error": "0 0 5px rgba(255, 51, 51, 0.5)",
      },
      // Custom animations
      animation: {
        blink: "blink 1s step-end infinite",
        typing: "typing 3.5s steps(40, end), blink-caret 0.75s step-end infinite",
        glitch: "glitch 0.3s ease-in-out",
        scanline: "scanline 8s linear infinite",
        pulse: "pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },
      keyframes: {
        blink: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0" },
        },
        typing: {
          from: { width: "0" },
          to: { width: "100%" },
        },
        "blink-caret": {
          "from, to": { borderColor: "transparent" },
          "50%": { borderColor: "#33ff00" },
        },
        glitch: {
          "0%": { transform: "translate(0)" },
          "20%": { transform: "translate(-2px, 2px)" },
          "40%": { transform: "translate(-2px, -2px)" },
          "60%": { transform: "translate(2px, 2px)" },
          "80%": { transform: "translate(2px, -2px)" },
          "100%": { transform: "translate(0)" },
        },
        scanline: {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100vh)" },
        },
      },
      // Box shadow for terminal windows
      boxShadow: {
        terminal: "0 0 10px rgba(51, 255, 0, 0.1)",
        "terminal-active": "0 0 20px rgba(51, 255, 0, 0.2)",
      },
    },
  },
  plugins: [],
};

export default config;
