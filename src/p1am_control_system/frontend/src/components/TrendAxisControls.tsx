import React from "react";
import type { AxisRange } from "../lib/trendAxis";

/**
 * Compact Y-axis range control for the trend plots: an "Auto Y" toggle plus
 * min/max number inputs (enabled only in manual mode). Shared across every
 * trend so the operator can rescale any plot the same way.
 */
export const TrendAxisControls: React.FC<{
  value: AxisRange;
  onChange: (next: AxisRange) => void;
  unit?: string;
}> = ({ value, onChange, unit }) => {
  const num = (raw: string): number => {
    const n = Number.parseFloat(raw);
    return Number.isFinite(n) ? n : 0;
  };
  return (
    <div
      className="trend-axis-controls"
      style={{
        display: "flex",
        alignItems: "center",
        gap: "0.5rem",
        fontSize: "0.68rem",
        color: "var(--text-secondary)",
        flexWrap: "wrap",
      }}
    >
      <label
        style={{ display: "flex", alignItems: "center", gap: "0.25rem", cursor: "pointer" }}
        title="Auto-fit the Y axis to the visible data"
      >
        <input
          type="checkbox"
          checked={value.auto}
          onChange={(e) => onChange({ ...value, auto: e.target.checked })}
        />
        Auto Y
      </label>
      <label style={{ display: "flex", alignItems: "center", gap: "0.2rem", opacity: value.auto ? 0.45 : 1 }}>
        min
        <input
          type="number"
          disabled={value.auto}
          value={value.min}
          onChange={(e) => onChange({ ...value, min: num(e.target.value) })}
          style={{ width: "4.5rem", fontSize: "0.68rem", padding: "0.1rem 0.25rem" }}
        />
      </label>
      <label style={{ display: "flex", alignItems: "center", gap: "0.2rem", opacity: value.auto ? 0.45 : 1 }}>
        max
        <input
          type="number"
          disabled={value.auto}
          value={value.max}
          onChange={(e) => onChange({ ...value, max: num(e.target.value) })}
          style={{ width: "4.5rem", fontSize: "0.68rem", padding: "0.1rem 0.25rem" }}
        />
      </label>
      {unit && <span style={{ color: "var(--text-muted)" }}>{unit}</span>}
    </div>
  );
};
