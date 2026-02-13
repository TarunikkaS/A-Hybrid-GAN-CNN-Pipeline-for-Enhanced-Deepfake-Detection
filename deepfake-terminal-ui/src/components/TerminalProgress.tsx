"use client";

import clsx from "clsx";

interface AsciiProgressProps {
  value: number; // 0-100
  width?: number; // Character width
  label?: string;
  showPercentage?: boolean;
}

export function AsciiProgress({
  value,
  width = 30,
  label,
  showPercentage = true,
}: AsciiProgressProps) {
  const clampedValue = Math.min(100, Math.max(0, value));
  const filledChars = Math.round((clampedValue / 100) * width);
  const emptyChars = width - filledChars;

  const progressBar = `[${"█".repeat(filledChars)}${"░".repeat(emptyChars)}]`;

  return (
    <div className="font-mono text-terminal-primary text-glow">
      {label && (
        <div className="mb-1 text-xs uppercase tracking-wider text-terminal-muted">
          {label}
        </div>
      )}
      <div className="flex items-center gap-2">
        <span className="whitespace-pre">{progressBar}</span>
        {showPercentage && (
          <span className={clsx(
            "text-sm",
            clampedValue >= 50 ? "text-terminal-error" : "text-terminal-primary"
          )}>
            {clampedValue.toFixed(1)}%
          </span>
        )}
      </div>
    </div>
  );
}

interface StatusBadgeProps {
  status: "real" | "fake" | "processing" | "error";
  large?: boolean;
}

export function StatusBadge({ status, large = false }: StatusBadgeProps) {
  const config = {
    real: {
      text: "[ REAL ]",
      color: "text-terminal-primary border-terminal-primary",
      glow: "text-glow",
    },
    fake: {
      text: "[ FAKE ]",
      color: "text-terminal-error border-terminal-error",
      glow: "text-glow-error",
    },
    processing: {
      text: "[ ... ]",
      color: "text-terminal-secondary border-terminal-secondary animate-pulse",
      glow: "text-glow-amber",
    },
    error: {
      text: "[ ERR ]",
      color: "text-terminal-error border-terminal-error",
      glow: "text-glow-error",
    },
  };

  const { text, color, glow } = config[status];

  return (
    <span
      className={clsx(
        "font-mono uppercase tracking-widest border px-3 py-1",
        color,
        glow,
        large ? "text-2xl px-6 py-3" : "text-sm"
      )}
    >
      {text}
    </span>
  );
}

interface TerminalMetricProps {
  label: string;
  value: string | number;
  unit?: string;
  status?: "ok" | "warning" | "error";
}

export function TerminalMetric({
  label,
  value,
  unit,
  status = "ok",
}: TerminalMetricProps) {
  const statusColors = {
    ok: "text-terminal-primary",
    warning: "text-terminal-secondary",
    error: "text-terminal-error",
  };

  return (
    <div className="font-mono">
      <div className="text-xs uppercase tracking-wider text-terminal-muted mb-1">
        {label}
      </div>
      <div className={clsx("text-2xl font-bold", statusColors[status])}>
        {value}
        {unit && <span className="text-sm ml-1 text-terminal-muted">{unit}</span>}
      </div>
    </div>
  );
}
