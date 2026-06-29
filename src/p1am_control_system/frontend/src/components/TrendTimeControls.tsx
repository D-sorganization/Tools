import React, { useEffect, useRef, useState } from "react";
import {
  type TimeUnit,
  SELECTABLE_TIME_UNITS,
  unitLabel,
  toSeconds,
  fromSeconds,
  windowUnit,
  clampWindow,
  formatWindow,
  MAX_WINDOW_SECONDS,
} from "../lib/trendTime";

/** Trim a number for display in the input: "5", "1.5", not "5.0000". */
function trimNum(n: number): string {
  return `${Number(n.toFixed(4))}`;
}

/**
 * Time-window (X-axis) control: a numeric text box plus a unit dropdown
 * (sec / min / hr). The operator types a value and picks a unit; the parent
 * gets the window in seconds (clamped to the supported range). Shared across
 * every trend so the time basis is set the same way everywhere.
 */
export const TrendTimeControls: React.FC<{
  value: number; // window in seconds
  onChange: (seconds: number) => void;
}> = ({ value, onChange }) => {
  const [unit, setUnit] = useState<TimeUnit>(() => windowUnit(value));
  const [text, setText] = useState<string>(() => trimNum(fromSeconds(value, windowUnit(value))));
  const focused = useRef(false);

  // Resync the box when the window changes externally (but never while the user
  // is mid-edit, so we don't clobber typing).
  useEffect(() => {
    if (!focused.current) setText(trimNum(fromSeconds(value, unit)));
  }, [value, unit]);

  const commit = (raw: string, nextUnit: TimeUnit): void => {
    const parsed = Number.parseFloat(raw);
    if (!Number.isFinite(parsed) || parsed <= 0) return; // wait for a valid value
    onChange(clampWindow(toSeconds(parsed, nextUnit)));
  };

  return (
    <div
      style={{ display: "flex", alignItems: "center", gap: "0.3rem", fontSize: "0.68rem" }}
      title={`Time window shown on the plot (max ${formatWindow(MAX_WINDOW_SECONDS)})`}
    >
      <span style={{ color: "var(--text-muted)", textTransform: "uppercase" }}>Window</span>
      <input
        type="number"
        min={0}
        step="any"
        value={text}
        onFocus={() => {
          focused.current = true;
        }}
        onBlur={() => {
          focused.current = false;
          // Normalize the box to the actual (clamped) window.
          setText(trimNum(fromSeconds(value, unit)));
        }}
        onChange={(e) => {
          setText(e.target.value);
          commit(e.target.value, unit);
        }}
        style={{ width: "4.5rem", fontSize: "0.68rem", padding: "0.1rem 0.25rem" }}
        aria-label="Time window value"
      />
      <select
        value={unit}
        onChange={(e) => {
          const nextUnit = e.target.value as TimeUnit;
          setUnit(nextUnit);
          // Keep the same window length; just re-express it in the new unit.
          setText(trimNum(fromSeconds(value, nextUnit)));
        }}
        style={{ fontSize: "0.68rem", padding: "0.1rem 0.2rem" }}
        aria-label="Time window unit"
      >
        {SELECTABLE_TIME_UNITS.map((u) => (
          <option key={u} value={u}>
            {unitLabel(u)}
          </option>
        ))}
      </select>
    </div>
  );
};
