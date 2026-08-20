import { useEffect, useRef, useState } from "react";

import type { ScalarScatterData } from "../model/scalarEnsembleContract";
import { BUTTON_CLASS } from "./variationUi";

const PALETTE = ["#38bdf8", "#f59e0b", "#ef4444", "#a78bfa", "#34d399"];
const LEFT = 72;
const RIGHT = 18;
const TOP = 16;
const BOTTOM = 48;
const TICK_COUNT = 5;

const bounds = (values: readonly number[]): readonly [number, number] => {
  if (!values.length) return [-1, 1];
  const low = Math.min(...values); const high = Math.max(...values);
  const pad = Math.max((high - low) * 0.08, Math.max(Math.abs(low), 1) * 1e-6);
  return [low - pad, high + pad];
};

const zoomedBounds = (
  values: readonly number[], zoom: number,
): readonly [number, number] => {
  const [low, high] = bounds(values);
  const middle = (low + high) / 2;
  const span = (high - low) / zoom;
  return [middle - span / 2, middle + span / 2];
};

const ticks = (low: number, high: number): readonly number[] => Array.from(
  { length: TICK_COUNT }, (_, index) => low + (high - low) * index / (TICK_COUNT - 1),
);

const tickText = (value: number): string => new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 3,
}).format(value);

const cohortLabel = (cohort: string): string => cohort.split("_")
  .map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join(" ");

function drawGrid(
  context: CanvasRenderingContext2D, width: number, height: number,
  xBounds: readonly [number, number], yBounds: readonly [number, number],
): void {
  context.font = "12px sans-serif";
  context.fillStyle = "#94a3b8";
  context.strokeStyle = "#1e293b";
  ticks(...xBounds).forEach((value, index) => {
    const x = LEFT + index * (width - LEFT - RIGHT) / (TICK_COUNT - 1);
    context.beginPath(); context.moveTo(x, TOP); context.lineTo(x, height - BOTTOM); context.stroke();
    context.textAlign = "center"; context.fillText(tickText(value), x, height - BOTTOM + 18);
  });
  ticks(...yBounds).forEach((value, index) => {
    const y = height - BOTTOM - index * (height - TOP - BOTTOM) / (TICK_COUNT - 1);
    context.beginPath(); context.moveTo(LEFT, y); context.lineTo(width - RIGHT, y); context.stroke();
    context.textAlign = "right"; context.fillText(tickText(value), LEFT - 8, y + 4);
  });
}

function drawScatter(
  canvas: HTMLCanvasElement,
  scatter: ScalarScatterData<string>,
  zoom: number,
): void {
  const context = canvas.getContext("2d");
  if (!context) return;
  const width = canvas.width; const height = canvas.height;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#020617"; context.fillRect(0, 0, width, height);
  const [xLow, xHigh] = zoomedBounds(scatter.points.map(({ x }) => x), zoom);
  const [yLow, yHigh] = zoomedBounds(scatter.points.map(({ y }) => y), zoom);
  const xSpan = xHigh - xLow; const ySpan = yHigh - yLow;
  const cohorts = scatter.availability.by_cohort;
  const cohortIds = Object.keys(cohorts);
  drawGrid(context, width, height, [xLow, xHigh], [yLow, yHigh]);
  context.strokeStyle = "#64748b";
  context.strokeRect(LEFT, TOP, width - LEFT - RIGHT, height - TOP - BOTTOM);
  scatter.points.forEach((point) => {
    const x = LEFT + (point.x - xLow) / xSpan * (width - LEFT - RIGHT);
    const y = height - BOTTOM - (point.y - yLow) / ySpan * (height - TOP - BOTTOM);
    if (x < LEFT || x > width - RIGHT || y < TOP || y > height - BOTTOM) return;
    context.beginPath(); context.arc(x, y, 2.5, 0, Math.PI * 2);
    context.fillStyle = PALETTE[Math.max(0, cohortIds.indexOf(point.cohort)) % PALETTE.length];
    context.fill();
  });
}

export function ScalarEnsembleScatter({ scatter, label }: {
  readonly scatter: ScalarScatterData<string>; readonly label: string;
}): JSX.Element {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [zoom, setZoom] = useState(1);
  useEffect(() => setZoom(1), [scatter]);
  useEffect(() => {
    if (canvas.current) drawScatter(canvas.current, scatter, zoom);
  }, [scatter, zoom]);
  const [xLow, xHigh] = zoomedBounds(scatter.points.map(({ x }) => x), zoom);
  const [yLow, yHigh] = zoomedBounds(scatter.points.map(({ y }) => y), zoom);
  const cohorts = Object.entries(scatter.availability.by_cohort);
  return <section aria-label={`${label} scatter`} className="space-y-2">
    <div className="flex flex-wrap items-center gap-2">
      <button type="button" className={BUTTON_CLASS}
        onClick={() => setZoom((value) => Math.min(8, value * 1.25))}>Zoom In</button>
      <button type="button" className={BUTTON_CLASS}
        onClick={() => setZoom((value) => Math.max(0.5, value / 1.25))}>Zoom Out</button>
      <button type="button" className={BUTTON_CLASS} onClick={() => setZoom(1)}>Auto Fit</button>
      <span role="status" className="text-xs text-slate-400">{Math.round(zoom * 100)}%</span>
    </div>
    <canvas ref={canvas} width="900" height="420" role="img"
      aria-label={`${label}: ${scatter.x_variable.label} versus ${scatter.y_variable.label}; ${scatter.points.length} paired finite points`}
      className="h-auto w-full rounded border border-slate-800" />
    <p className="text-center text-xs text-slate-400">
      {scatter.x_variable.label} range {tickText(xLow)} to {tickText(xHigh)} {
        scatter.x_variable.unit}; {scatter.y_variable.label} range {tickText(yLow)} to {
        tickText(yHigh)} {scatter.y_variable.unit}
    </p>
    <ul aria-label={`${label} cohort legend`}
      className="flex flex-wrap justify-center gap-x-4 gap-y-1 text-xs text-slate-300">{
      cohorts.map(([cohort, availability], index) => <li key={cohort} className="flex items-center gap-1.5">
        <span aria-hidden="true" className="inline-block h-2.5 w-2.5 rounded-full"
          style={{ backgroundColor: PALETTE[index % PALETTE.length] }} />
        {cohortLabel(cohort)} {availability.paired_finite}
      </li>)}</ul>
    <p className="text-center text-xs text-slate-400">
      {scatter.x_variable.label} [{scatter.x_variable.unit}] versus {scatter.y_variable.label} [{scatter.y_variable.unit}]
    </p>
  </section>;
}
