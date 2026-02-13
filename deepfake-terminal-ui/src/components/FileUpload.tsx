"use client";

import { useCallback, useState } from "react";
import { Upload } from "lucide-react";
import clsx from "clsx";

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  accept?: string;
  maxSize?: number; // in MB
  label?: string;
}

export function FileUpload({
  onFileSelect,
  accept = "image/*,video/*",
  maxSize = 50,
  label = "UPLOAD FILE",
}: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      setError(null);

      const file = e.dataTransfer.files[0];
      if (file) {
        validateAndSelect(file);
      }
    },
    [maxSize, onFileSelect]
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setError(null);
      const file = e.target.files?.[0];
      if (file) {
        validateAndSelect(file);
      }
    },
    [maxSize, onFileSelect]
  );

  const validateAndSelect = (file: File) => {
    // Check file size
    if (file.size > maxSize * 1024 * 1024) {
      setError(`[ERR] FILE SIZE EXCEEDS ${maxSize}MB LIMIT`);
      return;
    }

    onFileSelect(file);
  };

  return (
    <div className="w-full">
      <label
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={clsx(
          "flex flex-col items-center justify-center w-full h-48",
          "border-2 border-dashed cursor-pointer",
          "transition-all duration-200",
          isDragging
            ? "border-terminal-primary bg-terminal-dimmed"
            : "border-terminal-border hover:border-terminal-muted",
          error && "border-terminal-error"
        )}
      >
        <div className="flex flex-col items-center justify-center pt-5 pb-6">
          <Upload
            className={clsx(
              "w-10 h-10 mb-4",
              isDragging ? "text-terminal-primary animate-pulse" : "text-terminal-muted"
            )}
            strokeWidth={1.5}
          />
          <p className="mb-2 text-sm text-terminal-primary text-glow">
            <span className="font-semibold">$ {label}</span>
          </p>
          <p className="text-xs text-terminal-muted">
            {">"} DRAG & DROP OR CLICK TO SELECT
          </p>
          <p className="text-xs text-terminal-muted mt-1">
            {">"} MAX SIZE: {maxSize}MB
          </p>
        </div>
        <input
          type="file"
          className="hidden"
          accept={accept}
          onChange={handleFileInput}
        />
      </label>

      {error && (
        <div className="mt-2 text-terminal-error text-xs text-glow-error">
          {error}
        </div>
      )}
    </div>
  );
}
