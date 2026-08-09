import { useEffect, useRef, useState } from "react";

import type { ScalarScatterData } from "../model/scalarEnsembleContract";
import { BUTTON_CLASS } from "./variationUi";

const PALETTE = ["#38bdf8", "#f59e0b", "#ef4444", "#a78bfa", "#34d399"];

const bounds = (values: readonly number[]): readonly [number, number] => {
  if (!values.length) return [-1, 1];
  const low = Math.min(...values); const high = Math.max(...values);
  const pad = Math.max((high - low) * 0.08, Math.max(Math.abs(low), 1) * 1e-6);
  return [low - pad, high + pad];
};

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
  const [xLow, xHigh] = bounds(scatter.points.map(({ x }) => x));
  const [yLow, yHigh] = bounds(scatter.points.map(({ y }) => y));
  const xMid = (xLow + xHigh) / 2; const yMid = (yLow + yHigh) / 2;
  const xSpan = (xHigh - xLow) / zoom; const ySpan = (yHigh - yLow) / zoom;
  const cohorts = scatter.availability.by_cohort;
  const cohortIds = Object.keys(cohorts);
  context.strokeStyle = "#475569"; context.strokeRect(50, 15, width - 65, height - 55);
  scatter.points.forEach((point) => {
    const x = 50 + (point.x - (xMid - xSpan / 2)) / xSpan * (width - 65);
    const y = height - 40 - (point.y - (yMid - ySpan / 2)) / ySpan * (height - 55);
    if (x < 50 || x > width - 15 || y < 15 || y > height - 40) return;
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
      {scatter.x_variable.label} [{scatter.x_variable.unit}] versus {scatter.y_variable.label} [{scatter.y_variable.unit}]
    </p>
  </section>;
}
