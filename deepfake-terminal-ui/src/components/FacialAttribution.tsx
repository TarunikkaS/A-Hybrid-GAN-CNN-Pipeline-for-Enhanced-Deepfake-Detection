"use client";

interface FacialRegion {
  name: string;
  delta: number;
  occludedProb: number;
}

interface FacialAttributionProps {
  regions: FacialRegion[];
  originalProb: number;
  prediction: "real" | "fake";
}

export function FacialAttribution({
  regions,
  originalProb,
  prediction,
}: FacialAttributionProps) {
  // Sort by absolute delta
  const sortedRegions = [...regions].sort(
    (a, b) => Math.abs(b.delta) - Math.abs(a.delta)
  );

  const strongest = sortedRegions[0];

  return (
    <div className="font-mono">
      {/* Strongest Indicator */}
      {strongest && (
        <div className="mb-6 p-4 border border-terminal-border">
          <div className="text-xs uppercase tracking-wider text-terminal-muted mb-2">
            // STRONGEST INDICATOR
          </div>
          <div className="flex items-center justify-between">
            <span
              className={`text-xl font-bold uppercase ${
                strongest.delta > 0
                  ? "text-terminal-error text-glow-error"
                  : "text-terminal-primary text-glow"
              }`}
            >
              {strongest.name.replace("_", " ")}
            </span>
            <span
              className={`text-lg ${
                strongest.delta > 0
                  ? "text-terminal-error"
                  : "text-terminal-primary"
              }`}
            >
              Δ = {strongest.delta > 0 ? "+" : ""}
              {strongest.delta.toFixed(4)}
            </span>
          </div>
          <div className="text-xs text-terminal-muted mt-2">
            {">"} WHEN OCCLUDED: P(FAKE) {originalProb.toFixed(4)} →{" "}
            {strongest.occludedProb.toFixed(4)}
          </div>
        </div>
      )}

      {/* Top 3 Regions */}
      <div className="text-xs uppercase tracking-wider text-terminal-muted mb-3">
        // TOP 3 CONTRIBUTING REGIONS
      </div>
      <div className="grid grid-cols-3 gap-4">
        {sortedRegions.slice(0, 3).map((region, idx) => (
          <div
            key={region.name}
            className="border border-terminal-border p-3"
          >
            <div className="text-terminal-muted text-xs mb-1">
              #{idx + 1}
            </div>
            <div className="text-sm uppercase text-terminal-primary">
              {region.name.replace("_", " ")}
            </div>
            <div
              className={`text-lg font-bold ${
                region.delta > 0
                  ? "text-terminal-error"
                  : "text-terminal-primary"
              }`}
            >
              {region.delta > 0 ? "+" : ""}
              {region.delta.toFixed(4)}
            </div>
            <div className="text-xs text-terminal-muted mt-1">
              {Math.abs(region.delta) > 0.05
                ? "HIGH IMPACT"
                : "LOW IMPACT"}
            </div>
          </div>
        ))}
      </div>

      {/* Legend */}
      <div className="mt-6 border-t border-terminal-border pt-4 text-xs text-terminal-muted">
        <div className="mb-2 uppercase tracking-wider">// HOW TO READ:</div>
        <div>{">"} Δ (DELTA): CHANGE IN P(FAKE) WHEN REGION IS OCCLUDED</div>
        <div>
          {">"} <span className="text-terminal-error">+Δ</span>: REGION
          CONTRIBUTES TO FAKE EVIDENCE
        </div>
        <div>
          {">"} <span className="text-terminal-primary">-Δ</span>: REGION
          CONTRIBUTES TO REAL EVIDENCE
        </div>
      </div>
    </div>
  );
}
