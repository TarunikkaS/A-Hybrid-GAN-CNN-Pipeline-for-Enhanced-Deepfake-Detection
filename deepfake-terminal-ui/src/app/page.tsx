"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Terminal, 
  Upload, 
  Scan, 
  AlertTriangle, 
  CheckCircle2, 
  XCircle,
  Video,
  Image as ImageIcon,
  Zap,
  Activity
} from "lucide-react";
import {
  TerminalWindow,
  TerminalButton,
  AsciiProgress,
  StatusBadge,
  TerminalMetric,
  FileUpload,
  FacialAttribution,
  AsciiArt,
  ASCII_LOGO,
} from "@/components";
import { analyzeImage, analyzeVideo, checkHealth } from "@/lib/api";
import { AnalysisResult, VideoAnalysisResult, FacialRegionResult } from "@/lib/types";

type Tab = "image" | "video";
type AnalysisState = "idle" | "uploading" | "analyzing" | "complete" | "error";

export default function Home() {
  // State
  const [activeTab, setActiveTab] = useState<Tab>("image");
  const [analysisState, setAnalysisState] = useState<AnalysisState>("idle");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [imageResult, setImageResult] = useState<AnalysisResult | null>(null);
  const [videoResult, setVideoResult] = useState<VideoAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showGradcam, setShowGradcam] = useState(true);

  // Handle file selection
  const handleFileSelect = useCallback((file: File) => {
    setSelectedFile(file);
    setError(null);
    setImageResult(null);
    setVideoResult(null);
    
    // Create preview URL
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setAnalysisState("idle");
  }, []);

  // Analyze image
  const handleAnalyzeImage = useCallback(async () => {
    if (!selectedFile) return;
    
    setAnalysisState("analyzing");
    setError(null);
    
    try {
      console.log("[DEBUG] Starting image analysis...");
      const result = await analyzeImage(selectedFile, {
        includeGradcam: showGradcam,
        includeRegions: true,
      });
      console.log("[DEBUG] Analysis result:", result);
      setImageResult(result);
      setAnalysisState("complete");
    } catch (err) {
      console.error("[DEBUG] Analysis error:", err);
      const errorMessage = err instanceof Error ? err.message : "Analysis failed";
      setError(errorMessage);
      setAnalysisState("error");
    }
  }, [selectedFile, showGradcam]);

  // Analyze video
  const handleAnalyzeVideo = useCallback(async () => {
    if (!selectedFile) return;
    
    setAnalysisState("analyzing");
    setError(null);
    
    try {
      const result = await analyzeVideo(selectedFile, {
        sampleEvery: 15,
        maxFrames: 40,
        method: "mean",
      });
      setVideoResult(result);
      setAnalysisState("complete");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Video analysis failed");
      setAnalysisState("error");
    }
  }, [selectedFile]);

  // Reset
  const handleReset = useCallback(() => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setImageResult(null);
    setVideoResult(null);
    setError(null);
    setAnalysisState("idle");
  }, []);

  return (
    <main className="min-h-screen bg-terminal-bg p-4 md:p-8">
      {/* Header */}
      <header className="mb-8">
        <div className="text-center mb-6">
          <AsciiArt art={ASCII_LOGO} className="hidden md:block mx-auto" />
          <h1 className="md:hidden text-2xl font-bold text-terminal-primary text-glow uppercase tracking-widest">
            DEEPFAKE DETECTOR
          </h1>
        </div>
        
        <div className="flex items-center justify-center gap-4 text-terminal-muted text-sm">
          <span className="flex items-center gap-2">
            <Terminal className="w-4 h-4" />
            <span>v1.0.0</span>
          </span>
          <span className="text-terminal-border">|</span>
          <span className="flex items-center gap-2">
            <Zap className="w-4 h-4" />
            <span>XCEPTION-NET</span>
          </span>
          <span className="text-terminal-border">|</span>
          <span className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-terminal-primary animate-pulse" />
            <span className="text-terminal-primary">ONLINE</span>
          </span>
        </div>
      </header>

      {/* Tab Navigation */}
      <div className="flex justify-center gap-4 mb-8">
        <button
          onClick={() => { setActiveTab("image"); handleReset(); }}
          className={`flex items-center gap-2 px-6 py-3 border font-mono uppercase tracking-wide transition-all ${
            activeTab === "image"
              ? "border-terminal-primary bg-terminal-primary text-terminal-bg"
              : "border-terminal-border text-terminal-muted hover:border-terminal-muted"
          }`}
        >
          <ImageIcon className="w-4 h-4" />
          <span>IMAGE</span>
        </button>
        <button
          onClick={() => { setActiveTab("video"); handleReset(); }}
          className={`flex items-center gap-2 px-6 py-3 border font-mono uppercase tracking-wide transition-all ${
            activeTab === "video"
              ? "border-terminal-primary bg-terminal-primary text-terminal-bg"
              : "border-terminal-border text-terminal-muted hover:border-terminal-muted"
          }`}
        >
          <Video className="w-4 h-4" />
          <span>VIDEO</span>
        </button>
      </div>

      {/* Main Content Grid */}
      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Panel: Upload & Preview */}
        <TerminalWindow 
          title={activeTab === "image" ? "IMAGE INPUT" : "VIDEO INPUT"}
          status={analysisState === "analyzing" ? "processing" : analysisState === "error" ? "error" : "ok"}
        >
          {!selectedFile ? (
            <FileUpload
              onFileSelect={handleFileSelect}
              accept={activeTab === "image" ? "image/*" : "video/*"}
              label={activeTab === "image" ? "UPLOAD IMAGE" : "UPLOAD VIDEO"}
              maxSize={activeTab === "image" ? 10 : 100}
            />
          ) : (
            <div className="space-y-4">
              {/* Preview */}
              <div className="border border-terminal-border p-2">
                {activeTab === "image" && previewUrl && (
                  <img
                    src={previewUrl}
                    alt="Preview"
                    className="w-full h-auto max-h-64 object-contain"
                  />
                )}
                {activeTab === "video" && previewUrl && (
                  <video
                    src={previewUrl}
                    controls
                    className="w-full h-auto max-h-64"
                  />
                )}
              </div>

              {/* File Info */}
              <div className="text-xs text-terminal-muted font-mono">
                <div>{">"} FILE: {selectedFile.name}</div>
                <div>{">"} SIZE: {(selectedFile.size / 1024 / 1024).toFixed(2)} MB</div>
                <div>{">"} TYPE: {selectedFile.type}</div>
              </div>

              {/* Options */}
              {activeTab === "image" && (
                <label className="flex items-center gap-2 text-sm text-terminal-muted cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showGradcam}
                    onChange={(e) => setShowGradcam(e.target.checked)}
                    className="w-4 h-4 accent-terminal-primary"
                  />
                  <span>INCLUDE GRAD-CAM HEATMAP</span>
                </label>
              )}

              {/* Action Buttons */}
              <div className="flex gap-4">
                <TerminalButton
                  onClick={activeTab === "image" ? handleAnalyzeImage : handleAnalyzeVideo}
                  loading={analysisState === "analyzing"}
                  disabled={analysisState === "analyzing"}
                >
                  <span className="flex items-center gap-2">
                    <Scan className="w-4 h-4" />
                    ANALYZE
                  </span>
                </TerminalButton>
                <TerminalButton
                  variant="secondary"
                  onClick={handleReset}
                  disabled={analysisState === "analyzing"}
                >
                  RESET
                </TerminalButton>
              </div>

              {/* Error */}
              {error && (
                <div className="flex items-center gap-2 text-terminal-error text-sm">
                  <XCircle className="w-4 h-4" />
                  <span>[ERR] {error}</span>
                </div>
              )}
            </div>
          )}
        </TerminalWindow>

        {/* Right Panel: Results */}
        <TerminalWindow
          title="ANALYSIS OUTPUT"
          status={
            analysisState === "complete"
              ? imageResult?.label === "FAKE" || videoResult?.verdict === "FAKE"
                ? "error"
                : "ok"
              : analysisState === "analyzing"
              ? "processing"
              : "ok"
          }
        >
          <AnimatePresence mode="wait">
            {analysisState === "idle" && !selectedFile && (
              <motion.div
                key="idle"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="h-64 flex items-center justify-center text-terminal-muted"
              >
                <div className="text-center">
                  <Terminal className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <div className="text-sm">AWAITING INPUT...</div>
                  <div className="text-xs mt-2">$ upload a file to begin analysis</div>
                </div>
              </motion.div>
            )}

            {analysisState === "idle" && selectedFile && (
              <motion.div
                key="ready"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="h-64 flex items-center justify-center text-terminal-secondary"
              >
                <div className="text-center">
                  <Upload className="w-12 h-12 mx-auto mb-4 animate-pulse" />
                  <div className="text-sm">FILE LOADED</div>
                  <div className="text-xs mt-2">$ click [ANALYZE] to start detection</div>
                </div>
              </motion.div>
            )}

            {analysisState === "analyzing" && (
              <motion.div
                key="analyzing"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="h-64 flex items-center justify-center"
              >
                <div className="text-center w-full max-w-md">
                  <div className="mb-6">
                    <Scan className="w-16 h-16 mx-auto text-terminal-primary animate-pulse" />
                  </div>
                  <div className="text-terminal-primary text-sm mb-4 animate-pulse">
                    ANALYZING {activeTab.toUpperCase()}...
                  </div>
                  <div className="space-y-2 text-xs text-terminal-muted text-left">
                    <div className="animate-pulse">{">"} Loading neural network weights...</div>
                    <div className="animate-pulse" style={{ animationDelay: "0.2s" }}>{">"} Preprocessing input data...</div>
                    <div className="animate-pulse" style={{ animationDelay: "0.4s" }}>{">"} Running inference...</div>
                    <div className="animate-pulse" style={{ animationDelay: "0.6s" }}>{">"} Computing attribution maps...</div>
                  </div>
                </div>
              </motion.div>
            )}

            {analysisState === "complete" && imageResult && (
              <motion.div
                key="image-result"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="space-y-6"
              >
                {/* Verdict */}
                <div className="text-center py-6 border border-terminal-border">
                  <div className="text-xs uppercase tracking-wider text-terminal-muted mb-3">
                    // VERDICT
                  </div>
                  <StatusBadge
                    status={imageResult.label === "FAKE" ? "fake" : "real"}
                    large
                  />
                  <div className="mt-4 text-terminal-muted text-sm">
                    CONFIDENCE: {(imageResult.confidence * 100).toFixed(2)}%
                  </div>
                </div>

                {/* Metrics */}
                <div className="grid grid-cols-2 gap-4">
                  <TerminalMetric
                    label="P(REAL)"
                    value={(imageResult.probabilities.real * 100).toFixed(2)}
                    unit="%"
                    status={imageResult.probabilities.real > 0.5 ? "ok" : "warning"}
                  />
                  <TerminalMetric
                    label="P(FAKE)"
                    value={(imageResult.probabilities.fake * 100).toFixed(2)}
                    unit="%"
                    status={imageResult.probabilities.fake > 0.5 ? "error" : "ok"}
                  />
                </div>

                {/* ASCII Progress */}
                <AsciiProgress
                  value={imageResult.probabilities.fake * 100}
                  label="FAKE PROBABILITY"
                  width={25}
                />

                {/* Processing Time */}
                <div className="text-xs text-terminal-muted">
                  {">"} PROCESSING TIME: {imageResult.processingTime.toFixed(3)}s
                </div>
              </motion.div>
            )}

            {analysisState === "complete" && videoResult && (
              <motion.div
                key="video-result"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="space-y-6"
              >
                {/* Verdict */}
                <div className="text-center py-6 border border-terminal-border">
                  <div className="text-xs uppercase tracking-wider text-terminal-muted mb-3">
                    // VIDEO VERDICT
                  </div>
                  <StatusBadge
                    status={videoResult.verdict === "FAKE" ? "fake" : "real"}
                    large
                  />
                  <div className="mt-4 text-terminal-muted text-sm">
                    SCORE: {(videoResult.score * 100).toFixed(2)}% | METHOD: {videoResult.method.toUpperCase()}
                  </div>
                </div>

                {/* Metrics */}
                <div className="grid grid-cols-3 gap-4">
                  <TerminalMetric
                    label="FRAMES"
                    value={videoResult.framesAnalyzed}
                  />
                  <TerminalMetric
                    label="SCORE"
                    value={(videoResult.score * 100).toFixed(1)}
                    unit="%"
                    status={videoResult.score > 0.5 ? "error" : "ok"}
                  />
                  <TerminalMetric
                    label="TIME"
                    value={videoResult.processingTime.toFixed(2)}
                    unit="s"
                  />
                </div>

                {/* Frame Timeline */}
                <div>
                  <div className="text-xs uppercase tracking-wider text-terminal-muted mb-3">
                    // SUSPICION TIMELINE
                  </div>
                  <div className="h-24 border border-terminal-border p-2 overflow-x-auto">
                    <div className="flex items-end gap-1 h-full">
                      {videoResult.frames.map((frame, idx) => (
                        <div
                          key={idx}
                          className="flex-shrink-0 w-2"
                          style={{ height: `${frame.pFake * 100}%` }}
                        >
                          <div
                            className={`w-full h-full ${
                              frame.pFake > 0.5
                                ? "bg-terminal-error"
                                : "bg-terminal-primary"
                            }`}
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="flex justify-between text-xs text-terminal-muted mt-1">
                    <span>FRAME 0</span>
                    <span>FRAME {videoResult.frames[videoResult.frames.length - 1]?.frameIndex || 0}</span>
                  </div>
                </div>
              </motion.div>
            )}

            {analysisState === "error" && (
              <motion.div
                key="error"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="h-64 flex items-center justify-center"
              >
                <div className="text-center text-terminal-error">
                  <XCircle className="w-16 h-16 mx-auto mb-4" />
                  <div className="text-lg mb-2">[ANALYSIS FAILED]</div>
                  <div className="text-sm text-terminal-muted">{error}</div>
                  <TerminalButton
                    variant="danger"
                    onClick={handleReset}
                    className="mt-4"
                  >
                    RETRY
                  </TerminalButton>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </TerminalWindow>
      </div>

      {/* Grad-CAM & Attribution Panel (only for images) */}
      {analysisState === "complete" && imageResult && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-7xl mx-auto mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6"
        >
          {/* Grad-CAM */}
          {imageResult.gradcamUrl && (
            <TerminalWindow title="GRAD-CAM HEATMAP">
              <div className="space-y-4">
                <div className="text-xs text-terminal-muted mb-2">
                  // REGIONS INFLUENCING PREDICTION
                </div>
                <div className="border border-terminal-border p-2">
                  <img
                    src={imageResult.gradcamUrl}
                    alt="Grad-CAM visualization"
                    className="w-full h-auto"
                  />
                </div>
                <div className="text-xs text-terminal-muted">
                  {">"} RED/YELLOW: HIGH ACTIVATION REGIONS
                  <br />
                  {">"} BLUE/GREEN: LOW ACTIVATION REGIONS
                </div>
              </div>
            </TerminalWindow>
          )}

          {/* Facial Attribution */}
          {imageResult.facialRegions && imageResult.facialRegions.length > 0 && (
            <TerminalWindow title="FACIAL REGION ANALYSIS">
              <FacialAttribution
                regions={imageResult.facialRegions}
                originalProb={imageResult.probabilities.fake}
                prediction={imageResult.label === "FAKE" ? "fake" : "real"}
              />
            </TerminalWindow>
          )}
        </motion.div>
      )}

      {/* Footer */}
      <footer className="mt-12 text-center text-terminal-muted text-xs">
        <div className="border-t border-terminal-border pt-4 max-w-7xl mx-auto">
          <div>{'─'.repeat(60)}</div>
          <div className="mt-2">
            DEEPFAKE DETECTOR v1.0.0 // XCEPTION NEURAL NETWORK // GRAD-CAM ATTRIBUTION
          </div>
          <div className="mt-1 text-terminal-border">
            {">"} USE AT YOUR OWN DISCRETION. THIS TOOL PROVIDES PROBABILISTIC ANALYSIS ONLY.
          </div>
        </div>
      </footer>
    </main>
  );
}
