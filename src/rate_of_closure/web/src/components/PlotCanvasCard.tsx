import { useEffect, useRef, useState } from "react";

import type { PlotData } from "../model/plotspec";

const PALETTE = ["#38bdf8", "#fbbf24", "#34d399", "#f472b6", "#a78bfa"];
const ZOOM_STEP = 1.25;

export type LegendPosition = "hidden" | "outside_right" | "inside_top_left" | "inside_top_right";

interface Props {
  data: PlotData;
  label: string;
  selected: boolean;
  onSelect: () => void;
  onCanvas: (canvas: HTMLCanvasElement | null) => void;
}

function fittedRange(values: number[], zoom: number): [number, number] {
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (min === max) {
    min -= 0.5;
    max += 0.5;
  }
  const center = (min + max) / 2;
  const halfRange = ((max - min) * 1.1) / (2 * zoom);
  return [center - halfRange, center + halfRange];
}

function drawLegend(
  ctx: CanvasRenderingContext2D,
  data: PlotData,
  position: LegendPosition,
  width: number,
): void {
  if (position === "hidden") return;
  const x = position === "outside_right" || position === "inside_top_right"
    ? width - (position === "outside_right" ? 170 : 185)
    : 76;
  const y = position === "outside_right" ? 52 : 46;
  data.series.forEach((series, index) => {
    ctx.fillStyle = PALETTE[index % PALETTE.length];
    ctx.fillRect(x, y + index * 18 - 8, 12, 3);
    ctx.fillText(series.label, x + 18, y + index * 18);
  });
}

function drawPlot(
  canvas: HTMLCanvasElement,
  data: PlotData,
  zoom: number,
  legend: LegendPosition,
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  const margin = { left: 64, right: legend === "outside_right" ? 190 : 20, top: 42, bottom: 46 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const [xMin, xMax] = fittedRange(data.x, zoom);
  const [yMin, yMax] = fittedRange(data.series.flatMap((series) => series.values), zoom);
  const sx = (value: number): number => margin.left + ((value - xMin) / (xMax - xMin)) * plotW;
  const sy = (value: number): number => margin.top + plotH - ((value - yMin) / (yMax - yMin)) * plotH;

  ctx.strokeStyle = "rgba(148, 163, 184, 0.35)";
  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px sans-serif";
  ctx.lineWidth = 1;
  for (let index = 0; index <= 5; index += 1) {
    const xValue = xMin + ((xMax - xMin) * index) / 5;
    const yValue = yMin + ((yMax - yMin) * index) / 5;
    const x = sx(xValue);
    const y = sy(yValue);
    ctx.beginPath();
    ctx.moveTo(x, margin.top);
    ctx.lineTo(x, margin.top + plotH);
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + plotW, y);
    ctx.stroke();
    ctx.textAlign = "center";
    ctx.fillText(xValue.toPrecision(3), x, margin.top + plotH + 16);
    ctx.textAlign = "right";
    ctx.fillText(yValue.toPrecision(3), margin.left - 6, y + 4);
  }
  data.series.forEach((series, seriesIndex) => {
    ctx.strokeStyle = PALETTE[seriesIndex % PALETTE.length];
    ctx.fillStyle = PALETTE[seriesIndex % PALETTE.length];
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    data.x.forEach((xValue, index) => {
      const x = sx(xValue);
      const y = sy(series.values[index]);
      if (data.spec.kind === "scatter") {
        ctx.moveTo(x + 2.5, y);
        ctx.arc(x, y, 2.5, 0, 2 * Math.PI);
      } else if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    if (data.spec.kind === "scatter") ctx.fill();
    else ctx.stroke();
  });
  drawLegend(ctx, data, legend, width);
  ctx.fillStyle = "#cbd5e1";
  ctx.font = "12px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(data.xLabel, margin.left + plotW / 2, height - 8);
  ctx.save();
  ctx.translate(14, margin.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(data.yLabel, 0, 0);
  ctx.restore();
  ctx.font = "bold 13px sans-serif";
  ctx.fillText(data.spec.title, margin.left + plotW / 2, 18);
}

export function PlotCanvasCard({ data, label, selected, onSelect, onCanvas }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [zoom, setZoom] = useState(1);
  const [legend, setLegend] = useState<LegendPosition>("outside_right");
  useEffect(() => {
    onCanvas(canvasRef.current);
    return () => onCanvas(null);
  }, [onCanvas]);
  useEffect(() => {
    if (canvasRef.current) drawPlot(canvasRef.current, data, zoom, legend);
  }, [data, zoom, legend]);
  const changeZoom = (factor: number): void => {
    setZoom((current) => Math.max(0.2, Math.min(20, current * factor)));
  };
  const buttonClass = "rounded border border-slate-700 px-2 py-1 text-xs text-slate-200 hover:border-sky-400";
  return (
    <article
      role="group"
      aria-label={`${label} plot controls`}
      className={`min-w-0 rounded-lg border p-3 ${selected ? "border-sky-400/70" : "border-slate-800"}`}
      onFocus={onSelect}
    >
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <strong className="mr-auto text-sm text-slate-200">{label}</strong>
        <button type="button" className={buttonClass} title="Magnify this plot around its fitted center."
          onClick={() => changeZoom(ZOOM_STEP)}>Zoom In</button>
        <button type="button" className={buttonClass} title="Show a wider range around this plot's fitted center."
          onClick={() => changeZoom(1 / ZOOM_STEP)}>Zoom Out</button>
        <button type="button" className={buttonClass} title="Recompute readable axis limits from all visible data."
          onClick={() => setZoom(1)}>Auto Fit</button>
        <output className="w-12 text-right text-xs tabular-nums text-sky-300">{Math.round(zoom * 100)}%</output>
        <label className="text-xs text-slate-300">
          Legend
          <select
            aria-label="Legend position"
            title="Place the legend outside the data, move it inside, or hide it."
            value={legend}
            onChange={(event) => setLegend(event.target.value as LegendPosition)}
            className="ml-1 rounded border border-slate-700 bg-slate-900 px-1 py-1"
          >
            <option value="outside_right">Outside Right</option>
            <option value="inside_top_right">Inside Top Right</option>
            <option value="inside_top_left">Inside Top Left</option>
            <option value="hidden">Hidden</option>
          </select>
        </label>
      </div>
      <canvas
        ref={canvasRef}
        width={860}
        height={420}
        role="img"
        tabIndex={0}
        aria-label={`${label} plot`}
        title="Use the mouse wheel to zoom; use Auto Fit to restore all data."
        onClick={onSelect}
        onWheel={(event) => {
          event.preventDefault();
          changeZoom(event.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP);
        }}
        className="h-auto min-h-[180px] w-full rounded-lg bg-slate-950/60 sm:min-h-0"
      />
    </article>
  );
}
