"use client";

import { ReactNode } from "react";
import clsx from "clsx";

interface TerminalWindowProps {
  title: string;
  children: ReactNode;
  className?: string;
  status?: "ok" | "error" | "warning" | "processing";
}

export function TerminalWindow({
  title,
  children,
  className,
  status = "ok",
}: TerminalWindowProps) {
  const statusIndicator = {
    ok: "[OK]",
    error: "[ERR]",
    warning: "[WARN]",
    processing: "[...]",
  };

  const statusColor = {
    ok: "text-terminal-primary",
    error: "text-terminal-error",
    warning: "text-terminal-secondary",
    processing: "text-terminal-muted animate-blink",
  };

  return (
    <div
      className={clsx(
        "border border-terminal-border bg-terminal-bg",
        "shadow-terminal",
        className
      )}
    >
      {/* Title Bar */}
      <div className="border-b border-terminal-border px-4 py-2 flex items-center justify-between bg-terminal-dimmed">
        <div className="flex items-center gap-2">
          <span className="text-terminal-muted">┌──</span>
          <span className="text-terminal-primary text-glow uppercase tracking-wider text-sm font-bold">
            {title}
          </span>
          <span className="text-terminal-muted">──┐</span>
        </div>
        <span className={clsx("text-xs font-mono", statusColor[status])}>
          {statusIndicator[status]}
        </span>
      </div>

      {/* Content */}
      <div className="p-4">{children}</div>

      {/* Footer */}
      <div className="border-t border-terminal-border px-4 py-1 text-terminal-muted text-xs">
        └{'─'.repeat(40)}┘
      </div>
    </div>
  );
}
