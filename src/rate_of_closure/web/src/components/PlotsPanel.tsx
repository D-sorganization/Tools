/**
 * Plots section (epic #4120 V1): built-in advanced plots, a simplified
 * custom builder (X / Y selects from the shared data catalog), canvas
 * line / scatter rendering with axis labels and units, PNG / CSV / JSON
 * export, and plot-definition import / export using the same JSON
 * schema as the desktop app — definitions travel both ways.
 */

import { useCallback, useMemo, useRef, useState } from "react";

import { type ImpactScenario } from "../model/impact";
import {
  axisLabel,
  supportedByCategory,
} from "../model/plotcatalog";
import {
  BUILTIN_PLOTS,
  plotDataCsv,
  plotDataJson,
  specFromJson,
  specToJson,
  type PlotSimulationExecutor,
} from "../model/plotspec";
import { PlotCanvasCard } from "./PlotCanvasCard";
import { usePlotsWorkspace } from "./usePlotsWorkspace";

interface Props {
  scenario: ImpactScenario;
  loftDeg: number;
  executeSimulation?: PlotSimulationExecutor;
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

export function PlotsPanel({ scenario, loftDeg, executeSimulation }: Props) {
  const canvasRefs = useRef(new Map<number, HTMLCanvasElement>());
  const fileRef = useRef<HTMLInputElement | null>(null);
  const initialPlot = useMemo(
    () => ({
      id: 1,
      label: "Closure Sweep",
      spec: BUILTIN_PLOTS[0].make(0.06),
    }),
    [],
  );
  const {
    addPlot,
    computed,
    context,
    plots,
    removeSelected,
    selectedId,
    setSelectedId,
    setWorkspaceError,
    workspaceError,
  } = usePlotsWorkspace({
    initialPlot,
    scenario,
    loftDeg,
    executeSimulation,
  });
  const [builtin, setBuiltin] = useState(BUILTIN_PLOTS[0].name);

  // Custom builder state (series categories only — the guided sweep
  // builder is a desktop feature for now).
  const seriesVariables = useMemo(
    () => CUSTOM_CATEGORIES.flatMap((c) => supportedByCategory(c)),
    [],
  );
  const [customX, setCustomX] = useState("swing.time_s");
  const [customY, setCustomY] = useState("swing.speed_mps");
  const [customKind, setCustomKind] = useState<"line" | "scatter">("line");

  const selected = plots.find((p) => p.id === selectedId) ?? plots[0];
  const selectedComputed = computed.find((entry) => entry.plot.id === selectedId);
  const data = selectedComputed?.data ?? null;
  const shownError = workspaceError;

  const addBuiltin = (): void => {
    const entry = BUILTIN_PLOTS.find((b) => b.name === builtin);
    if (!entry) return;
    const duration =
      context?.run.swing[context.run.swing.length - 1]?.t ?? 0.06;
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
      setWorkspaceError(String(exc));
    }
  };

  const exportPng = (): void => {
    canvasRefs.current.get(selectedId)?.toBlob((blob) => {
      if (blob) download("plot.png", blob);
    });
  };

  const registerCanvas = useCallback((id: number, canvas: HTMLCanvasElement | null): void => {
    if (canvas) canvasRefs.current.set(id, canvas);
    else canvasRefs.current.delete(id);
  }, []);

  const importDefinition = (file: File): void => {
    void file.text().then((text) => {
      try {
        const spec = specFromJson(JSON.parse(text));
        addPlot(spec.title || spec.x_key, spec);
      } catch (exc) {
        setWorkspaceError(String(exc));
      }
    });
  };

  const button =
    "rounded-lg border border-slate-700 bg-slate-900/70 px-3 py-1.5 text-xs " +
    "font-medium text-slate-200 transition-all hover:border-sky-400/60 " +
    "hover:text-sky-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500";
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
                    : "border-slate-800 bg-slate-900/50 text-slate-300 hover:border-slate-600") +
                  " focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
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
        aria-label="Plot workspace"
        className="order-first grid min-w-0 gap-4 rounded-xl border border-slate-800/80 bg-slate-900/60 p-4 shadow-lg shadow-black/20 lg:order-none 2xl:grid-cols-2"
      >
        {computed.map((entry) => entry.data ? (
          <PlotCanvasCard
            key={entry.plot.id}
            data={entry.data}
            label={entry.plot.label}
            selected={entry.plot.id === selectedId}
            onSelect={() => setSelectedId(entry.plot.id)}
            onCanvas={(canvas) => registerCanvas(entry.plot.id, canvas)}
            notice={entry.error && entry.retained
              ? `Recompute failed; prior accepted plot retained: ${entry.error}`
              : entry.error}
          />
        ) : entry.pending ? (
          <p key={entry.plot.id} role="status" className="text-sm text-slate-400">
            Computing {entry.plot.label}…
          </p>
        ) : (
          <p key={entry.plot.id} role="alert" className="text-sm text-rose-300">
            {entry.plot.label}: {entry.error}
          </p>
        ))}
      </section>
    </div>
  );
}
