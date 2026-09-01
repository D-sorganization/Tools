import { useEffect, useId, useMemo, useRef, useState } from "react";

import {
  histogramBinAtData,
  navigatePlotSelection,
  nearestSeriesPoint,
  planPlotInspection,
  type PlotInspectionPlan,
  type PlotNavigation,
  type PlotSelection,
} from "../model/plotPointInspector";
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
  notice?: string | null;
}

interface CanvasGeometry {
  margin: { left: number; right: number; top: number; bottom: number };
  plotW: number;
  plotH: number;
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  projected: readonly (readonly (readonly [number, number])[])[];
}

function axisCoordinate(value: number, logarithmic: boolean): number {
  return logarithmic ? Math.log10(value) : value;
}

function fittedRange(
  values: readonly number[], zoom: number, logarithmic: boolean,
): [number, number] {
  const transformed = values
    .filter((value) => Number.isFinite(value) && (!logarithmic || value > 0))
    .map((value) => axisCoordinate(value, logarithmic));
  let min = Math.min(...transformed);
  let max = Math.max(...transformed);
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
  // ⚡ Bolt Optimization: Replace .forEach iterations with a standard for loop to eliminate closure allocation overhead in hot paths.
  for (let index = 0; index < data.series.length; index++) {
    const series = data.series[index];
    ctx.fillStyle = PALETTE[index % PALETTE.length];
    ctx.fillRect(x, y + index * 18 - 8, 12, 3);
    ctx.fillText(series.label, x + 18, y + index * 18);
  }
}

function drawPlot(
  canvas: HTMLCanvasElement,
  data: PlotData,
  plan: PlotInspectionPlan | null,
  zoom: number,
  legend: LegendPosition,
  selection: PlotSelection | null,
): CanvasGeometry | null {
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  const margin = { left: 64, right: legend === "outside_right" ? 190 : 20, top: 42, bottom: 46 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const xValues = plan?.kind === "histogram"
    ? [plan.bins[0].lower, plan.bins[plan.bins.length - 1].upper]
    : [...data.x];
  const yValues = plan?.kind === "histogram"
    ? [0, ...plan.bins.map((item) => item.count)]
    : data.series.flatMap((series) => [...series.values]);
  const [xMin, xMax] = fittedRange(xValues, zoom, data.spec.x_log);
  const [yMin, yMax] = fittedRange(yValues, zoom, data.spec.y_log);
  const sx = (value: number): number => margin.left +
    ((axisCoordinate(value, data.spec.x_log) - xMin) / (xMax - xMin)) * plotW;
  const sy = (value: number): number => margin.top + plotH -
    ((axisCoordinate(value, data.spec.y_log) - yMin) / (yMax - yMin)) * plotH;

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
    const xLabelValue = data.spec.x_log ? 10 ** xValue : xValue;
    const yLabelValue = data.spec.y_log ? 10 ** yValue : yValue;
    ctx.fillText(xLabelValue.toPrecision(3), x, margin.top + plotH + 16);
    ctx.textAlign = "right";
    ctx.fillText(yLabelValue.toPrecision(3), margin.left - 6, y + 4);
  }
  const projected: Array<Array<readonly [number, number]>> = [];
  const renderSeries = plan?.kind === "series" ? plan.series : data.series;
  // ⚡ Bolt Optimization: Replace .forEach iterations with a standard for loop to eliminate closure allocation overhead in hot paths.
  for (let seriesIndex = 0; seriesIndex < renderSeries.length; seriesIndex++) {
    const series = renderSeries[seriesIndex];
    ctx.strokeStyle = PALETTE[seriesIndex % PALETTE.length];
    ctx.fillStyle = PALETTE[seriesIndex % PALETTE.length];
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    const points: Array<readonly [number, number]> = [];
    for (let index = 0; index < data.x.length; index++) {
      const xValue = data.x[index];
      const x = sx(xValue);
      const y = sy(series.values[index]);
      points.push([x, y]);
      if (data.spec.kind === "scatter") {
        ctx.moveTo(x + 2.5, y);
        ctx.arc(x, y, 2.5, 0, 2 * Math.PI);
      } else if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    if (data.spec.kind === "scatter") ctx.fill();
    else ctx.stroke();
    projected.push(points);
  }
  if (plan?.kind === "histogram") {
    ctx.fillStyle = PALETTE[0];
    for (let index = 0; index < plan.bins.length; index++) {
      const item = plan.bins[index];
      const left = sx(item.lower);
      const right = sx(item.upper);
      const top = sy(item.count);
      const bottom = sy(0);
      ctx.globalAlpha = 0.85;
      ctx.fillRect(left + 0.5, top, Math.max(0, right - left - 1), bottom - top);
      ctx.globalAlpha = 1;
    }
  }
  if (selection?.kind === "series" && plan?.kind === "series") {
    const [x, y] = projected[selection.seriesIndex][selection.rawIndex];
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, 2 * Math.PI);
    ctx.fillStyle = "#f8fafc";
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#0f172a";
    ctx.stroke();
  } else if (selection?.kind === "histogram" && plan?.kind === "histogram") {
    const item = plan.bins[selection.binIndex];
    const left = sx(item.lower);
    const right = sx(item.upper);
    const top = sy(item.count);
    const bottom = sy(0);
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#f8fafc";
    ctx.strokeRect(left + 1, top, Math.max(0, right - left - 2), bottom - top);
  }
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
  return { margin, plotW, plotH, xMin, xMax, yMin, yMax, projected };
}

function inspectionStatus(
  data: PlotData,
  plan: PlotInspectionPlan | null,
  selection: PlotSelection | null,
  inspectionError: string | null,
): string {
  if (plan === null) return `Exact inspection unavailable: ${inspectionError}.`;
  if (selection?.kind === "series" && plan.kind === "series") {
    const series = plan.series[selection.seriesIndex];
    return `Series ${series.label}; source point ${selection.rawIndex + 1}/${plan.rawCount}; ` +
      `${data.xLabel} ${plan.x[selection.rawIndex].toPrecision(6)}; ` +
      `${data.yLabel} ${series.values[selection.rawIndex].toPrecision(6)}.`;
  }
  if (selection?.kind === "histogram" && plan.kind === "histogram") {
    const item = plan.bins[selection.binIndex];
    return `Histogram bin ${item.index + 1}/${plan.bins.length}; ` +
      `${data.xLabel} [${item.lower.toPrecision(6)}, ${item.upper.toPrecision(6)}${
        item.index === plan.bins.length - 1 ? "]" : ")"
      }; count ${item.count}.`;
  }
  return plan.kind === "histogram"
    ? "No histogram bin selected. Click a bar; use Left, Right, Home, End, or Escape."
    : "No exact point selected. Click within 12 pixels; use arrow keys, Home, End, or Escape.";
}

export function PlotCanvasCard({ data, label, selected, onSelect, onCanvas, notice }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const geometryRef = useRef<CanvasGeometry | null>(null);
  const statusRef = useRef<HTMLParagraphElement | null>(null);
  const statusId = useId();
  const [zoom, setZoom] = useState(1);
  const [legend, setLegend] = useState<LegendPosition>("outside_right");
  const inspection = useMemo(
    () => {
      try {
        const plan = planPlotInspection(data.spec.kind, data.x, data.series);
        if (data.spec.x_log && plan.x.some((value) => value <= 0)) {
          throw new RangeError("logarithmic x evidence must be positive");
        }
        if (data.spec.y_log && plan.series.some((item) =>
          item.values.some((value) => value <= 0))) {
          throw new RangeError("logarithmic y evidence must be positive");
        }
        return { plan, error: null };
      } catch (error) {
        return { plan: null, error: String(error).slice(0, 512) };
      }
    },
    [data],
  );
  const { plan } = inspection;
  const [selectedPoint, setSelectedPoint] = useState<{
    plan: PlotInspectionPlan;
    selection: PlotSelection;
  } | null>(null);
  const selection = plan !== null && selectedPoint?.plan === plan
    ? selectedPoint.selection : null;
  useEffect(() => {
    onCanvas(canvasRef.current);
    return () => onCanvas(null);
  }, [onCanvas]);
  useEffect(() => {
    if (canvasRef.current) {
      geometryRef.current = drawPlot(canvasRef.current, data, plan, zoom, legend, selection);
    }
  }, [data, plan, zoom, legend, selection]);
  const changeZoom = (factor: number): void => {
    setZoom((current) => Math.max(0.2, Math.min(20, current * factor)));
  };
  const adoptSelection = (value: PlotSelection | null): void => {
    if (value !== null && plan !== null) setSelectedPoint({ plan, selection: value });
    else setSelectedPoint(null);
    requestAnimationFrame(() => {
      const status = statusRef.current;
      if (status === null) return;
      const rect = status.getBoundingClientRect();
      if (rect.top < 0 || rect.bottom > window.innerHeight) {
        status.scrollIntoView({ behavior: "auto", block: "nearest" });
      }
    });
  };
  const pickPoint = (event: React.MouseEvent<HTMLCanvasElement>): void => {
    onSelect();
    const canvas = canvasRef.current;
    const geometry = geometryRef.current;
    if (!canvas || !geometry || plan === null) return;
    const rect = canvas.getBoundingClientRect();
    if (!(rect.width > 0 && rect.height > 0)) return;
    const pointer = [event.clientX - rect.left, event.clientY - rect.top] as const;
    if (plan.kind === "series") {
      const projected = geometry.projected.map((series) => series.map(([x, y]) => [
        (x * rect.width) / canvas.width,
        (y * rect.height) / canvas.height,
      ]));
      const value = nearestSeriesPoint(plan, projected, pointer);
      if (value !== null) adoptSelection(value);
      return;
    }
    const xPixel = (pointer[0] * canvas.width) / rect.width;
    const yPixel = (pointer[1] * canvas.height) / rect.height;
    const xAxis = geometry.xMin + ((xPixel - geometry.margin.left) / geometry.plotW) *
      (geometry.xMax - geometry.xMin);
    const yAxis = geometry.yMax - ((yPixel - geometry.margin.top) / geometry.plotH) *
      (geometry.yMax - geometry.yMin);
    const x = data.spec.x_log ? 10 ** xAxis : xAxis;
    const y = data.spec.y_log ? 10 ** yAxis : yAxis;
    const value = histogramBinAtData(plan, x, y);
    if (value !== null) adoptSelection(value);
  };
  const navigate = (event: React.KeyboardEvent<HTMLCanvasElement>): void => {
    const commands: Partial<Record<string, PlotNavigation>> = {
      ArrowLeft: "previous", ArrowRight: "next", ArrowUp: "up", ArrowDown: "down",
      Home: "home", End: "end", Escape: "clear",
    };
    const command = commands[event.key];
    if (!command) return;
    event.preventDefault();
    onSelect();
    if (plan !== null) adoptSelection(navigatePlotSelection(plan, selection, command));
  };
  const buttonClass = "rounded border border-slate-700 px-2 py-1 text-xs text-slate-200 hover:border-sky-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500";
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
        <span aria-label="Zoom level" className="w-12 text-right text-xs tabular-nums text-sky-300">
          {Math.round(zoom * 100)}%
        </span>
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
        aria-describedby={statusId}
        aria-keyshortcuts="ArrowLeft ArrowRight ArrowUp ArrowDown Home End Escape"
        title="Click exact plotted evidence to inspect it. Arrow keys navigate; Escape clears. The mouse wheel zooms."
        onClick={pickPoint}
        onKeyDown={navigate}
        onWheel={(event) => {
          event.preventDefault();
          changeZoom(event.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP);
        }}
        className="h-auto min-h-[180px] w-full rounded-lg bg-slate-950/60 focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-300 sm:min-h-0"
      />
      <p ref={statusRef} id={statusId} role="status" aria-live="polite"
        className="mt-2 text-xs text-sky-200">
        {inspectionStatus(data, plan, selection, inspection.error)}
      </p>
      {notice ? <p role="alert" className="mt-1 text-xs text-rose-300">{notice}</p> : null}
    </article>
  );
}
