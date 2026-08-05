/**
 * Plots section (epic #4120 V1): built-in advanced plots, a simplified
 * custom builder (X / Y selects from the shared data catalog), canvas
 * line / scatter rendering with axis labels and units, PNG / CSV / JSON
 * export, and plot-definition import / export using the same JSON
 * schema as the desktop app — definitions travel both ways.
 */

import { useEffect, useMemo, useRef, useState } from "react";

import { type ImpactScenario } from "../model/impact";
import {
  axisLabel,
  supportedByCategory,
  type PlotContext,
} from "../model/plotcatalog";
import {
  BUILTIN_PLOTS,
  computePlotData,
  plotDataCsv,
  plotDataJson,
  specFromJson,
  specToJson,
  type PlotData,
  type PlotSpec,
} from "../model/plotspec";
import {
  runSimulation,
  type SimulationInput,
} from "../model/simulation";

const PALETTE = ["#38bdf8", "#fbbf24", "#34d399", "#f472b6", "#a78bfa"];

interface Props {
  scenario: ImpactScenario;
  loftDeg: number;
}

interface ManagedPlot {
  id: number;
  label: string;
  spec: PlotSpec;
}

const CUSTOM_CATEGORIES = ["Swing Sample", "Kinetics", "Flight"] as const;

function download(name: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

function drawPlot(canvas: HTMLCanvasElement, data: PlotData): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  const margin = { left: 64, right: 16, top: 36, bottom: 46 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;

  const allY = data.series.flatMap((s) => s.values);
  const ys = allY.length ? allY : [0, 1];
  const xMin = Math.min(...data.x);
  let xMax = Math.max(...data.x);
  let yMin = Math.min(...ys);
  let yMax = Math.max(...ys);
  if (xMax === xMin) xMax = xMin + 1;
  if (yMax === yMin) yMax = yMin + 1;
  const pad = 0.05 * (yMax - yMin);
  yMin -= pad;
  yMax += pad;
  const sx = (x: number): number =>
    margin.left + ((x - xMin) / (xMax - xMin)) * plotW;
  const sy = (y: number): number =>
    margin.top + plotH - ((y - yMin) / (yMax - yMin)) * plotH;

  // Frame + grid + ticks
  ctx.strokeStyle = "rgba(148, 163, 184, 0.35)";
  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px sans-serif";
  ctx.lineWidth = 1;
  const ticks = 5;
  for (let i = 0; i <= ticks; i += 1) {
    const fx = xMin + ((xMax - xMin) * i) / ticks;
    const fy = yMin + ((yMax - yMin) * i) / ticks;
    const px = sx(fx);
    const py = sy(fy);
    ctx.beginPath();
    ctx.moveTo(px, margin.top);
    ctx.lineTo(px, margin.top + plotH);
    ctx.moveTo(margin.left, py);
    ctx.lineTo(margin.left + plotW, py);
    ctx.stroke();
    ctx.textAlign = "center";
    ctx.fillText(fx.toPrecision(3), px, margin.top + plotH + 16);
    ctx.textAlign = "right";
    ctx.fillText(fy.toPrecision(3), margin.left - 6, py + 4);
  }

  // Series
  data.series.forEach((series, index) => {
    ctx.strokeStyle = PALETTE[index % PALETTE.length];
    ctx.fillStyle = PALETTE[index % PALETTE.length];
    if (data.spec.kind === "scatter") {
      for (let i = 0; i < data.x.length; i += 1) {
        ctx.beginPath();
        ctx.arc(sx(data.x[i]), sy(series.values[i]), 2.2, 0, 2 * Math.PI);
        ctx.fill();
      }
    } else {
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      for (let i = 0; i < data.x.length; i += 1) {
        const px = sx(data.x[i]);
        const py = sy(series.values[i]);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
    // Legend
    ctx.textAlign = "left";
    ctx.fillText(
      series.label,
      margin.left + 10 + index * 150,
      margin.top - 8,
    );
  });

  // Axis labels + title
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
  ctx.fillText(data.spec.title, margin.left + plotW / 2, 16);
}

export function PlotsPanel({ scenario, loftDeg }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);
  const [plots, setPlots] = useState<ManagedPlot[]>(() => [
    { id: 1, label: "Closure Sweep", spec: BUILTIN_PLOTS[0].make(0.06) },
  ]);
  const [selectedId, setSelectedId] = useState(1);
  const [builtin, setBuiltin] = useState(BUILTIN_PLOTS[0].name);
  const [error, setError] = useState<string | null>(null);
  const nextId = useRef(2);

  // Custom builder state (series categories only — the guided sweep
  // builder is a desktop feature for now).
  const seriesVariables = useMemo(
    () => CUSTOM_CATEGORIES.flatMap((c) => supportedByCategory(c)),
    [],
  );
  const [customX, setCustomX] = useState("swing.time_s");
  const [customY, setCustomY] = useState("swing.speed_mps");
  const [customKind, setCustomKind] = useState<"line" | "scatter">("line");

  const context: PlotContext = useMemo(() => {
    const input: SimulationInput = {
      sourceKind: "manual",
      clubheadSpeedMph: scenario.clubheadSpeedMph,
      omegaDps: [0, 0, 0],
      loftDeg,
      impactOffsetToeMm: scenario.impactOffsetToeMm,
      impactOffsetHighMm: scenario.impactOffsetHighMm,
      planeYawDeg: 0,
      planeSideTiltDeg: -45,
      planeForwardTiltDeg: 0,
      impactTimeS: null,
      swingDurationS: 1.5,
    };
    return { scenario, input, run: runSimulation(input) };
  }, [scenario, loftDeg]);

  const selected = plots.find((p) => p.id === selectedId) ?? plots[0];
  const computed = useMemo((): {
    data: PlotData | null;
    error: string | null;
  } => {
    if (!selected) return { data: null, error: null };
    try {
      return { data: computePlotData(selected.spec, context), error: null };
    } catch (exc) {
      return { data: null, error: String(exc) };
    }
  }, [selected, context]);
  const data = computed.data;
  const shownError = error ?? computed.error;

  useEffect(() => {
    if (canvasRef.current && data) drawPlot(canvasRef.current, data);
  }, [data]);

  const addPlot = (label: string, spec: PlotSpec): void => {
    const id = nextId.current;
    nextId.current += 1;
    setPlots((list) => [...list, { id, label, spec }]);
    setSelectedId(id);
  };

  const addBuiltin = (): void => {
    const entry = BUILTIN_PLOTS.find((b) => b.name === builtin);
    if (!entry) return;
    const duration =
      context.run.swing[context.run.swing.length - 1]?.t ?? 0.06;
    addPlot(entry.label, entry.make(duration));
  };

  const addCustom = (): void => {
    try {
      addPlot(`Custom — ${axisLabel(customY)}`, {
        kind: customKind,
        x_key: customX,
        y_keys: [customY],
        series_key: null,
        title: `Custom Plot — ${axisLabel(customY)} vs ${axisLabel(customX)}`,
        x_log: false,
        y_log: false,
        x_start: null,
        x_stop: null,
        x_count: 25,
      });
    } catch (exc) {
      setError(String(exc));
    }
  };

  const removeSelected = (): void => {
    setPlots((list) => {
      const next = list.filter((p) => p.id !== selectedId);
      if (next.length) setSelectedId(next[next.length - 1].id);
      return next;
    });
  };

  const exportPng = (): void => {
    canvasRef.current?.toBlob((blob) => {
      if (blob) download("plot.png", blob);
    });
  };

  const importDefinition = (file: File): void => {
    void file.text().then((text) => {
      try {
        const spec = specFromJson(JSON.parse(text));
        addPlot(spec.title || spec.x_key, spec);
      } catch (exc) {
        setError(String(exc));
      }
    });
  };

  const button =
    "rounded-lg border border-slate-700 bg-slate-900/70 px-3 py-1.5 text-xs " +
    "font-medium text-slate-200 transition-all hover:border-sky-400/60 " +
    "hover:text-sky-300";
  const select =
    "w-full rounded-lg border border-slate-700 bg-slate-900/70 px-2 py-1.5 " +
    "text-sm text-slate-100";

  return (
    <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
      <section
        aria-label="Plot management"
        className="space-y-4 rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 shadow-lg shadow-black/20"
      >
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
          Plots
        </h2>
        <ul className="space-y-1" aria-label="Managed plots">
          {plots.map((plot) => (
            <li key={plot.id}>
              <button
                type="button"
                title="Select this plot to render it on the canvas."
                aria-pressed={plot.id === selectedId}
                onClick={() => setSelectedId(plot.id)}
                className={
                  "w-full rounded-lg border px-3 py-1.5 text-left text-sm " +
                  (plot.id === selectedId
                    ? "border-sky-400/60 bg-sky-500/10 text-sky-200"
                    : "border-slate-800 bg-slate-900/50 text-slate-300 hover:border-slate-600")
                }
              >
                {plot.label}
              </button>
            </li>
          ))}
        </ul>
        <div className="flex gap-2">
          <select
            className={select}
            value={builtin}
            aria-label="Built-in plot"
            title="Built-in advanced plots: sweeps, time series, and flight profiles."
            onChange={(e) => setBuiltin(e.target.value)}
          >
            {BUILTIN_PLOTS.map((entry) => (
              <option key={entry.name} value={entry.name}>
                {entry.label}
              </option>
            ))}
          </select>
          <button
            type="button"
            className={button}
            title="Add the selected built-in plot to the list."
            onClick={addBuiltin}
          >
            Add
          </button>
        </div>
        <div className="space-y-2 rounded-lg border border-slate-800 p-3">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Custom Plot
          </h3>
          <label className="block text-xs text-slate-400">
            X Variable
            <select
              className={select}
              value={customX}
              title="Horizontal-axis variable, from the shared data catalog."
              onChange={(e) => setCustomX(e.target.value)}
            >
              {seriesVariables.map((entry) => (
                <option key={entry.key} value={entry.key}>
                  {entry.category} — {entry.label} [{entry.unit}]
                </option>
              ))}
            </select>
          </label>
          <label className="block text-xs text-slate-400">
            Y Variable
            <select
              className={select}
              value={customY}
              title="Vertical-axis variable, from the shared data catalog."
              onChange={(e) => setCustomY(e.target.value)}
            >
              {seriesVariables.map((entry) => (
                <option key={entry.key} value={entry.key}>
                  {entry.category} — {entry.label} [{entry.unit}]
                </option>
              ))}
            </select>
          </label>
          <label className="block text-xs text-slate-400">
            Kind
            <select
              className={select}
              value={customKind}
              title="Line joins samples in order; Scatter draws one marker per sample."
              onChange={(e) =>
                setCustomKind(e.target.value as "line" | "scatter")
              }
            >
              <option value="line">Line</option>
              <option value="scatter">Scatter</option>
            </select>
          </label>
          <button
            type="button"
            className={button}
            title="Add the custom plot described above to the list."
            onClick={addCustom}
          >
            Add Custom Plot
          </button>
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            className={button}
            title="Remove the selected plot from the list."
            onClick={removeSelected}
          >
            Remove
          </button>
          <button
            type="button"
            className={button}
            title="Download the rendered plot as a PNG image."
            onClick={exportPng}
          >
            PNG
          </button>
          <button
            type="button"
            className={button}
            title="Download the plotted numbers as CSV."
            onClick={() =>
              data &&
              download(
                "plot_data.csv",
                new Blob([plotDataCsv(data)], { type: "text/csv" }),
              )
            }
          >
            Data CSV
          </button>
          <button
            type="button"
            className={button}
            title="Download the plotted numbers plus the definition as JSON."
            onClick={() =>
              data &&
              download(
                "plot_data.json",
                new Blob([plotDataJson(data)], { type: "application/json" }),
              )
            }
          >
            Data JSON
          </button>
          <button
            type="button"
            className={button}
            title="Download this plot's definition (.json); it reloads here or in the desktop app."
            onClick={() =>
              selected &&
              download(
                "plot_definition.json",
                new Blob([JSON.stringify(specToJson(selected.spec), null, 2)], {
                  type: "application/json",
                }),
              )
            }
          >
            Save Definition
          </button>
          <button
            type="button"
            className={button}
            title="Load a saved plot definition (.json) into the list."
            onClick={() => fileRef.current?.click()}
          >
            Load Definition
          </button>
          <input
            ref={fileRef}
            type="file"
            accept="application/json"
            className="hidden"
            aria-label="Plot definition file"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) importDefinition(file);
              e.target.value = "";
            }}
          />
        </div>
        {shownError ? (
          <p role="alert" className="text-xs text-rose-300">
            {shownError}
          </p>
        ) : null}
      </section>
      <section
        aria-label="Plot canvas"
        className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-4 shadow-lg shadow-black/20"
      >
        <canvas
          ref={canvasRef}
          width={860}
          height={520}
          className="h-auto w-full rounded-lg bg-slate-950/60"
          title="Rendered plot — export it with the PNG button."
        />
      </section>
    </div>
  );
}
