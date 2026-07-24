/**
 * Shared state types + pure helpers for the Data Explorer UI.
 *
 * The {@link DataExplorer} container owns all state and the API calls; the
 * panels are controlled views that render this state and emit changes. Keeping
 * the shapes and the pure transforms here (a) lets the container and panels
 * agree on one contract and (b) makes the data-shaping logic unit-testable
 * without React.
 */

import type {
  Column,
  CorrelationMethod,
  DatasetResponse,
  DerivedColumn,
  FilterSpec,
  ResampleSpec,
  SpectrumMethod,
  TrendlineKind,
  TrimSpec,
  WindowKind,
} from "../../api/explorerSchemas";
import type { NotificationType } from "../../types";

/** The HMI-wide toast callback, shared by every explorer panel. */
export type NotifyFn = (message: string, type: NotificationType) => void;

export type SourceMode = "historian" | "csv";

export type PlotKind =
  | "line"
  | "scatter"
  | "histogram"
  | "correlation"
  | "spectrum"
  | "pca";

/** Operator-entered historian query before it becomes a HistorianSource. */
export interface HistorianForm {
  tags: string[];
  start: string;
  end: string;
  maxPoints: number;
}

/** A browser-parsed CSV ready to send as an inline dataset source. */
export interface CsvSource {
  name: string;
  index: number[] | null;
  columns: Column[];
}

/** The processing pipeline applied when (re)building a dataset. */
export interface Pipeline {
  resample: ResampleSpec | null;
  filters: FilterSpec[];
  derived: DerivedColumn[];
  trim: TrimSpec | null;
}

/** Everything the Visualize + analysis panels need to render a chart. */
export interface PlotConfig {
  kind: PlotKind;
  /** Columns plotted on the line chart (multi-series). */
  columns: string[];
  /** Scatter axes. */
  xColumn: string;
  yColumn: string;
  logX: boolean;
  logY: boolean;
  grid: boolean;
  legend: boolean;
  trendline: TrendlineKind | "none";
  degree: number;
  histColumn: string;
  bins: number;
  spectrumColumn: string;
  spectrumMethod: SpectrumMethod;
  window: WindowKind;
  correlationMethod: CorrelationMethod;
  standardize: boolean;
}

/** The serializable analysis session persisted to localStorage. */
export interface ExplorerSession {
  sourceMode: SourceMode;
  historian: HistorianForm;
  pipeline: Pipeline;
  plot: PlotConfig;
}

export function defaultPipeline(): Pipeline {
  return { resample: null, filters: [], derived: [], trim: null };
}

export function defaultPlotConfig(): PlotConfig {
  return {
    kind: "line",
    columns: [],
    xColumn: "",
    yColumn: "",
    logX: false,
    logY: false,
    grid: true,
    legend: true,
    trendline: "none",
    degree: 2,
    histColumn: "",
    bins: 30,
    spectrumColumn: "",
    spectrumMethod: "fft",
    window: "hanning",
    correlationMethod: "pearson",
    standardize: true,
  };
}

/** Column names of a built dataset, in order. */
export function columnNames(dataset: DatasetResponse | null): string[] {
  return dataset ? dataset.columns.map((c) => c.name) : [];
}

/** Look up a column's raw values by name (nullable samples preserved). */
export function columnValues(
  dataset: DatasetResponse | null,
  name: string,
): Array<number | null> {
  const col = dataset?.columns.find((c) => c.name === name);
  return col ? col.values : [];
}

/**
 * Convert an epoch-millisecond index to seconds-relative-to-start, which makes
 * a readable time axis (raw epoch-ms ticks are unreadable). If the index is
 * already small (e.g. a synthetic 0..n-1 from a CSV without a time column) it
 * is returned unchanged.
 */
export function relativeSeconds(index: number[]): number[] {
  if (index.length === 0) return [];
  const first = index[0];
  // Heuristic: epoch-ms values are huge (> ~1e11); treat those as time.
  const looksLikeEpochMs = Math.abs(first) > 1e11;
  if (!looksLikeEpochMs) return index.slice();
  return index.map((t) => (t - first) / 1000);
}

/**
 * Build `[x, y]` line points from a dataset column, using the (relative-time)
 * index as x. Non-finite samples become `[x, NaN]` so the line breaks at gaps.
 */
export function linePoints(
  index: number[],
  values: Array<number | null>,
): [number, number][] {
  const x = relativeSeconds(index);
  const n = Math.min(x.length, values.length);
  const out: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    const v = values[i];
    out.push([x[i], v === null ? NaN : v]);
  }
  return out;
}

/**
 * Build `[x, y]` scatter points from two columns, dropping any pair where
 * either value is null/non-finite (so downstream extents stay finite).
 */
export function scatterPoints(
  xValues: Array<number | null>,
  yValues: Array<number | null>,
): [number, number][] {
  const n = Math.min(xValues.length, yValues.length);
  const out: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    const xv = xValues[i];
    const yv = yValues[i];
    if (xv === null || yv === null) continue;
    if (!Number.isFinite(xv) || !Number.isFinite(yv)) continue;
    out.push([xv, yv]);
  }
  return out;
}

/** Reconcile a saved plot config with the current dataset's columns. */
export function reconcilePlotColumns(
  plot: PlotConfig,
  names: string[],
): PlotConfig {
  if (names.length === 0) return plot;
  // ⚡ Bolt Optimization: Pre-compute a Set for O(1) lookups instead of O(N) array .includes()
  const namesSet = new Set(names);
  const has = (n: string) => namesSet.has(n);
  const first = names[0];
  const second = names[1] ?? names[0];
  const keptColumns = plot.columns.filter(has);
  return {
    ...plot,
    columns: keptColumns.length > 0 ? keptColumns : [first],
    xColumn: has(plot.xColumn) ? plot.xColumn : first,
    yColumn: has(plot.yColumn) ? plot.yColumn : second,
    histColumn: has(plot.histColumn) ? plot.histColumn : first,
    spectrumColumn: has(plot.spectrumColumn) ? plot.spectrumColumn : first,
  };
}
