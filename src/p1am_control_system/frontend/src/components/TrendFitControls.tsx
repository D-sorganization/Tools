import React from "react";
import { fitOptions } from "../lib/curveFit";

/**
 * Curve-fit method selector: a dropdown populated from the curveFit registry
 * (None + every registered method). Adding a method to the registry adds it
 * here automatically — no change needed in this component or the plots.
 */
export const TrendFitControls: React.FC<{
  value: string; // selected fit method id
  onChange: (methodId: string) => void;
}> = ({ value, onChange }) => (
  <div style={{ display: "flex", alignItems: "center", gap: "0.3rem", fontSize: "0.68rem" }}>
    <span style={{ color: "var(--text-muted)", textTransform: "uppercase" }}>Fit</span>
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{ fontSize: "0.68rem", padding: "0.1rem 0.2rem" }}
      aria-label="Curve fit method"
    >
      {fitOptions().map((o) => (
        <option key={o.id} value={o.id}>
          {o.label}
        </option>
      ))}
    </select>
  </div>
);
