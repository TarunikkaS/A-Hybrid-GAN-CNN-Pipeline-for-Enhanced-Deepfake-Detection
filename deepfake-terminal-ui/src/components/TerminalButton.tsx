"use client";

import { ButtonHTMLAttributes } from "react";
import clsx from "clsx";

interface TerminalButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "danger";
  size?: "sm" | "md" | "lg";
  loading?: boolean;
}

export function TerminalButton({
  children,
  variant = "primary",
  size = "md",
  loading = false,
  className,
  disabled,
  ...props
}: TerminalButtonProps) {
  const variants = {
    primary: "border-terminal-primary text-terminal-primary hover:bg-terminal-primary hover:text-terminal-bg",
    secondary: "border-terminal-secondary text-terminal-secondary hover:bg-terminal-secondary hover:text-terminal-bg",
    danger: "border-terminal-error text-terminal-error hover:bg-terminal-error hover:text-terminal-bg",
  };

  const sizes = {
    sm: "px-2 py-1 text-xs",
    md: "px-4 py-2 text-sm",
    lg: "px-6 py-3 text-base",
  };

  return (
    <button
      className={clsx(
        "font-mono uppercase tracking-wide border",
        "bg-transparent cursor-pointer",
        "transition-all duration-100",
        "disabled:opacity-50 disabled:cursor-not-allowed",
        variants[variant],
        sizes[size],
        loading && "animate-pulse",
        className
      )}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? (
        <span className="flex items-center gap-2">
          <span className="animate-blink">[</span>
          <span>PROCESSING</span>
          <span className="animate-blink">]</span>
        </span>
      ) : (
        <span>[ {children} ]</span>
      )}
    </button>
  );
}
