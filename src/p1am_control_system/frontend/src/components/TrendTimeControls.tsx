import React from "react";
import { TIME_WINDOW_PRESETS, formatWindow } from "../lib/trendTime";

/**
 * Compact time-window (X-axis) selector for the trend plots: one button per
 * preset window. Shared across the simple SVG trends so the operator can set
 * how much history each plot shows the same way.
 */
export const TrendTimeControls: React.FC<{
  value: number; // selected window in seconds
  onChange: (seconds: number) => void;
}> = ({ value, onChange }) => (
  <div
    style={{
      display: "flex",
      alignItems: "center",
      gap: "0.3rem",
      fontSize: "0.68rem",
      flexWrap: "wrap",
    }}
  >
    <span style={{ color: "var(--text-muted)", textTransform: "uppercase" }}>Window</span>
    {TIME_WINDOW_PRESETS.map((s) => (
      <button
        key={s}
        type="button"
        onClick={() => onChange(s)}
        className="btn"
        style={{
          padding: "0.1rem 0.4rem",
          fontSize: "0.68rem",
          border: "1px solid",
          borderColor: value === s ? "var(--accent-cyan)" : "var(--panel-border)",
          background: value === s ? "var(--cell-hover-bg)" : "var(--input-bg)",
          color: value === s ? "var(--accent-cyan)" : "var(--text-secondary)",
        }}
      >
        {formatWindow(s)}
      </button>
    ))}
  </div>
);
