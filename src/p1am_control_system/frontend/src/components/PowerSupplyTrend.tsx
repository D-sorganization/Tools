import React from "react";

/**
 * Compact dual-trace trend for the power-supply screen: measured current and
 * voltage coming back from the unit, plotted as a percent of their respective
 * full scales on a shared 0–100 % axis so both fit one clean chart. Live
 * engineering values are shown in the legend.
 *
 * Self-contained SVG (no chart lib): the parent accumulates a rolling sample
 * buffer from the live status and passes it in.
 */

export interface TrendSample {
  i: number; // measured current (A)
  v: number; // measured voltage (V)
}

interface Props {
  samples: TrendSample[];
  currentFullScale: number;
  voltageFullScale: number;
  windowSeconds: number;
}

const W = 600;
const H = 180;
const PAD_L = 34;
const PAD_R = 10;
const PAD_T = 10;
const PAD_B = 18;
const PLOT_W = W - PAD_L - PAD_R;
const PLOT_H = H - PAD_T - PAD_B;

const CURRENT_COLOR = "var(--color-success)";
const VOLTAGE_COLOR = "var(--accent-purple)";

function buildPath(
  values: number[],
  fullScale: number,
): string {
  if (values.length < 2 || fullScale <= 0) return "";
  const n = values.length;
  return values
    .map((val, idx) => {
      const x = PAD_L + (idx / (n - 1)) * PLOT_W;
      const frac = Math.max(0, Math.min(1, val / fullScale));
      const y = PAD_T + (1 - frac) * PLOT_H;
      return `${idx === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

export const PowerSupplyTrend: React.FC<Props> = ({
  samples,
  currentFullScale,
  voltageFullScale,
  windowSeconds,
}) => {
  const currentPath = buildPath(
    samples.map((s) => s.i),
    currentFullScale,
  );
  const voltagePath = buildPath(
    samples.map((s) => s.v),
    voltageFullScale,
  );
  const last = samples[samples.length - 1];

  return (
    <div className="ps-trend">
      <div className="ps-trend-legend">
        <span className="ps-trend-key">
          <span className="swatch" style={{ background: CURRENT_COLOR }} />
          Current
          <strong>{last ? `${last.i.toFixed(2)} A` : "—"}</strong>
        </span>
        <span className="ps-trend-key">
          <span className="swatch" style={{ background: VOLTAGE_COLOR }} />
          Voltage
          <strong>{last ? `${last.v.toFixed(2)} V` : "—"}</strong>
        </span>
        <span className="ps-trend-window">last {windowSeconds}s · % of full scale</span>
      </div>

      <svg
        className="ps-trend-svg"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        role="img"
        aria-label="Current and voltage trend"
      >
        {/* gridlines at 0/25/50/75/100 % */}
        {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
          const y = PAD_T + (1 - frac) * PLOT_H;
          return (
            <g key={frac}>
              <line
                x1={PAD_L}
                y1={y}
                x2={W - PAD_R}
                y2={y}
                className="ps-trend-grid"
              />
              <text x={PAD_L - 6} y={y + 3} className="ps-trend-axis" textAnchor="end">
                {frac * 100}
              </text>
            </g>
          );
        })}

        {samples.length < 2 ? (
          <text x={W / 2} y={H / 2} className="ps-trend-empty" textAnchor="middle">
            waiting for live data…
          </text>
        ) : (
          <>
            <path d={voltagePath} className="ps-trend-line" stroke={VOLTAGE_COLOR} />
            <path d={currentPath} className="ps-trend-line" stroke={CURRENT_COLOR} />
          </>
        )}
      </svg>
    </div>
  );
};
