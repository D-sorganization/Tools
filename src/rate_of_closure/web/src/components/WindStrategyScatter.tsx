import { useEffect, useId, useState } from "react";

import { BUTTON_CLASS, INPUT_CLASS } from "./variationUi";
import type { ScalarScatterData } from "../model/scalarEnsembleContract";
import type { WindOutcomeStatus } from "../model/windUncertainty";

const COLORS: Record<WindOutcomeStatus, string> = {
  completed: "#38bdf8", nonconverged: "#f59e0b", invalid: "#ef4444",
};
const COHORTS: WindOutcomeStatus[] = ["completed", "nonconverged", "invalid"];

const bounds = (values: number[]): [number, number] => {
  if (!values.length) return [-1, 1];
  // ⚡ Bolt Optimization: Calculate bounds dynamically in a single pass to avoid spreading large arrays
  // on the call stack, which causes "Maximum call stack size exceeded" errors and heavy GC pressure.
  let low = values[0] ?? 0;
  let high = values[0] ?? 0;
  for (let i = 1; i < values.length; i++) {
    const value = values[i] as number;
    if (value < low) low = value;
    if (value > high) high = value;
  }
  const padding = Math.max((high - low) * 0.08, Math.max(Math.abs(low), 1) * 1e-6);
  return [low - padding, high + padding];
};

const zoomBounds = ([low, high]: [number, number], zoom: number): [number, number] => {
  const center = (low + high) / 2; const halfSpan = (high - low) / (2 * zoom);
  return [center - halfSpan, center + halfSpan];
};

const ticks = (low: number, high: number): number[] => [low, (low + high) / 2, high];
const tickLabel = (value: number): string => {
  const magnitude = Math.abs(value);
  return magnitude !== 0 && (magnitude >= 10000 || magnitude < 0.001)
    ? value.toExponential(2) : Number(value.toPrecision(4)).toString();
};

export function WindStrategyScatter({ scatter }: {
  scatter: ScalarScatterData<WindOutcomeStatus>;
}): JSX.Element {
  const [zoom, setZoom] = useState(1);
  const [showLegend, setShowLegend] = useState(true);
  const [legendPosition, setLegendPosition] = useState<"top-right" | "bottom-right">("top-right");
  const clipId = `wind-scatter-clip-${useId().replace(/:/g, "")}`;
  useEffect(() => setZoom(1), [scatter]);
  const [x0, x1] = zoomBounds(bounds(scatter.points.map(({ x }) => x)), zoom);
  const [y0, y1] = zoomBounds(bounds(scatter.points.map(({ y }) => y)), zoom);
  const xTicks = ticks(x0, x1); const yTicks = ticks(y0, y1);
  const legendY = legendPosition === "top-right" ? 30 : 235;
  return <div className="space-y-2">
    <div className="flex flex-wrap items-center gap-2">
      <button type="button" className={BUTTON_CLASS} title="Zoom into the wind scatter"
        aria-label="Zoom in wind scatter" onClick={() => setZoom((value) => Math.min(value * 1.25, 8))}>Zoom In</button>
      <button type="button" className={BUTTON_CLASS} title="Zoom out of the wind scatter"
        aria-label="Zoom out wind scatter" onClick={() => setZoom((value) => Math.max(value / 1.25, 0.5))}>Zoom Out</button>
      <button type="button" className={BUTTON_CLASS} title="Restore data-fitted wind scatter bounds"
        onClick={() => setZoom(1)}>Auto Fit</button>
      <span role="status" aria-label="Wind scatter zoom" className="text-xs text-slate-400">
        {Math.round(zoom * 100)}%</span>
      <label className="flex items-center gap-1 text-xs text-slate-300">
        <input type="checkbox" checked={showLegend} title="Show or hide the outcome-cohort legend"
          onChange={(event) => setShowLegend(event.target.checked)} />Show wind scatter legend</label>
      <label className="text-xs text-slate-300">Legend position
        <select aria-label="Wind scatter legend position" title="Move the wind scatter legend"
          className={`${INPUT_CLASS} ml-1 inline-block w-auto`} value={legendPosition}
          onChange={(event) => setLegendPosition(event.target.value as typeof legendPosition)}>
          <option value="top-right">Top right</option><option value="bottom-right">Bottom right</option>
        </select></label>
    </div>
    <svg viewBox="0 0 640 340" role="img"
      aria-label={`Wind strategy scatter: ${scatter.x_variable.label} versus ${scatter.y_variable.label}`}
      className="h-auto w-full rounded border border-slate-800 bg-slate-950/60">
      <defs><clipPath id={clipId}><rect x="60" y="20" width="560" height="270" /></clipPath></defs>
      {xTicks.map((value, index) => {
        const x = 60 + index * 280;
        return <g key={`x-${value}`}><line data-testid="wind-scatter-gridline"
          x1={x} y1="20" x2={x} y2="290" stroke="#334155" strokeWidth="0.75" />
        <text data-testid="wind-scatter-tick" x={x} y="305" textAnchor="middle"
          fill="#94a3b8" fontSize="10">{tickLabel(value)}</text></g>;
      })}
      {yTicks.map((value, index) => {
        const y = 290 - index * 135;
        return <g key={`y-${value}`}><line data-testid="wind-scatter-gridline"
          x1="60" y1={y} x2="620" y2={y} stroke="#334155" strokeWidth="0.75" />
        <text data-testid="wind-scatter-tick" x="54" y={y + 3} textAnchor="end"
          fill="#94a3b8" fontSize="10">{tickLabel(value)}</text></g>;
      })}
      <line x1="60" y1="290" x2="620" y2="290" stroke="#64748b" />
      <line x1="60" y1="20" x2="60" y2="290" stroke="#64748b" />
      <g data-testid="wind-scatter-marks" clipPath={`url(#${clipId})`}>
      {scatter.points.map((point) => <circle key={point.row_id}
        cx={60 + (point.x - x0) / (x1 - x0) * 560}
        cy={290 - (point.y - y0) / (y1 - y0) * 270} r="4" fill={COLORS[point.cohort]}>
        <title>{`${point.row_id}: ${point.x}, ${point.y}`}</title></circle>)}</g>
      {showLegend && <g data-testid="wind-scatter-legend" transform={`translate(470 ${legendY})`}>
        <rect width="150" height="64" rx="5" fill="#0f172a" stroke="#475569" />
        {COHORTS.map((cohort, index) => <g key={cohort}
          transform={`translate(10 ${16 + index * 18})`}>
          <circle cx="4" cy="-3" r="4" fill={COLORS[cohort]} />
          <text x="14" fill="#cbd5e1" fontSize="11">{cohort}</text>
        </g>)}
      </g>}
      {!scatter.points.length && <text x="340" y="160" textAnchor="middle" fill="#94a3b8">
        No finite paired values for these axes</text>}
      <text x="340" y="325" textAnchor="middle" fill="#cbd5e1" fontSize="13">
        {scatter.x_variable.label} [{scatter.x_variable.unit}]</text>
      <text transform="translate(18 155) rotate(-90)" textAnchor="middle" fill="#cbd5e1" fontSize="13">
        {scatter.y_variable.label} [{scatter.y_variable.unit}]</text>
    </svg>
  </div>;
}
