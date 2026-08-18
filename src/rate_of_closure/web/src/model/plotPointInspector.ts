/** Immutable exact-point and derived-bin plans for managed plot inspection. */
export const MAX_PLOT_SAMPLES = 8_192;
export const MAX_PLOT_SERIES = 8;
export const MAX_PLOT_VERTICES = 8_192;
export const MAX_ABS_PLOT_VALUE = 1e12;
export const DEFAULT_PLOT_HIT_RADIUS_PX = 12;

export type PlotNavigation = "previous" | "next" | "up" | "down" | "home" | "end" | "clear";
export interface PlotSeries { readonly label: string; readonly values: readonly number[] }
export interface HistogramBin { readonly index: number; readonly lower: number; readonly upper: number; readonly count: number }
export interface PlotInspectionPlan {
  readonly kind: "series" | "histogram";
  readonly x: readonly number[];
  readonly series: readonly PlotSeries[];
  readonly bins: readonly HistogramBin[];
  readonly rawCount: number;
}
export type PlotSelection =
  | { readonly kind: "series"; readonly seriesIndex: number; readonly rawIndex: number }
  | { readonly kind: "histogram"; readonly binIndex: number };

function finite(value: unknown, field: string, pixel = false): number {
  const limit = pixel ? 1e9 : MAX_ABS_PLOT_VALUE;
  if (typeof value !== "number" || !Number.isFinite(value) || Math.abs(value) > limit) {
    throw new RangeError(`${field} must be a finite bounded number`);
  }
  return value;
}

function snapshotValues(value: unknown, field: string): readonly number[] {
  if (!Array.isArray(value) || value.length < 1 || value.length > MAX_PLOT_SAMPLES) {
    throw new RangeError(`plot evidence must contain 1..${MAX_PLOT_SAMPLES} samples`);
  }
  return Object.freeze(value.map((item) => finite(item, field)));
}

function histogramBins(x: readonly number[]): readonly HistogramBin[] {
  const count = Math.min(40, Math.max(10, Math.floor(x.length / 10)));
  let low = Math.min(...x);
  let high = Math.max(...x);
  if (low === high) { low -= 0.5; high += 0.5; }
  const width = (high - low) / count;
  const counts = new Array<number>(count).fill(0);
  for (const value of x) {
    const rawIndex = value === high ? count - 1 : Math.floor((value - low) / width);
    const index = Math.min(Math.max(rawIndex, 0), count - 1);
    counts[index] += 1;
  }
  return Object.freeze(counts.map((item, index) => Object.freeze({
    index, lower: low + index * width, upper: low + (index + 1) * width, count: item,
  })));
}

export function planPlotInspection(
  kind: unknown,
  xInput: unknown,
  seriesInput: unknown,
): PlotInspectionPlan {
  if (!["line", "scatter", "sweep", "histogram"].includes(String(kind))) {
    throw new RangeError("plot kind is not inspectable");
  }
  const x = snapshotValues(xInput, "x");
  if (!Array.isArray(seriesInput)) throw new RangeError("plot series must be a sized sequence");
  if (kind === "histogram") {
    if (seriesInput.length) throw new RangeError("histogram evidence must not contain y series");
    const bins = histogramBins(x);
    return Object.freeze({ kind: "histogram", x, series: Object.freeze([]), bins, rawCount: x.length });
  }
  if (seriesInput.length < 1 || seriesInput.length > MAX_PLOT_SERIES) {
    throw new RangeError(`plot must contain 1..${MAX_PLOT_SERIES} series`);
  }
  if (x.length * seriesInput.length > MAX_PLOT_VERTICES) {
    throw new RangeError(`plot exceeds ${MAX_PLOT_VERTICES} inspectable vertices`);
  }
  const series = Object.freeze(seriesInput.map((value): PlotSeries => {
    if (typeof value !== "object" || value === null) {
      throw new RangeError("plot series must contain label and values");
    }
    const item = value as Record<string, unknown>;
    if (typeof item.label !== "string" || item.label.length < 1 || item.label.length > 512) {
      throw new RangeError("plot series label must contain 1..512 characters");
    }
    const values = snapshotValues(item.values, "series values");
    if (values.length !== x.length) throw new RangeError("plot series values must align with x");
    return Object.freeze({ label: item.label, values });
  }));
  return Object.freeze({ kind: "series", x, series, bins: Object.freeze([]), rawCount: x.length });
}

function validSelection(plan: PlotInspectionPlan, selection: PlotSelection | null): PlotSelection | null {
  if (selection === null) return null;
  if (selection.kind === "series" && plan.kind === "series" &&
      Number.isSafeInteger(selection.seriesIndex) && Number.isSafeInteger(selection.rawIndex) &&
      selection.seriesIndex >= 0 && selection.seriesIndex < plan.series.length &&
      selection.rawIndex >= 0 && selection.rawIndex < plan.rawCount) return selection;
  if (selection.kind === "histogram" && plan.kind === "histogram" &&
      Number.isSafeInteger(selection.binIndex) && selection.binIndex >= 0 &&
      selection.binIndex < plan.bins.length) return selection;
  throw new RangeError("selection is outside the inspection plan");
}

export function navigatePlotSelection(
  plan: PlotInspectionPlan, currentInput: PlotSelection | null, command: PlotNavigation,
): PlotSelection | null {
  if (!["previous", "next", "up", "down", "home", "end", "clear"].includes(command)) {
    throw new RangeError("unknown plot navigation command");
  }
  const current = validSelection(plan, currentInput);
  if (command === "clear") return null;
  if (plan.kind === "histogram") {
    const index = current?.kind === "histogram" ? current.binIndex : null;
    if (command === "home") return { kind: "histogram", binIndex: 0 };
    if (command === "end") return { kind: "histogram", binIndex: plan.bins.length - 1 };
    if (index === null) return { kind: "histogram", binIndex: ["next", "down"].includes(command) ? 0 : plan.bins.length - 1 };
    const delta = ["next", "down"].includes(command) ? 1 : -1;
    return { kind: "histogram", binIndex: Math.min(Math.max(index + delta, 0), plan.bins.length - 1) };
  }
  const item = current?.kind === "series" ? current : null;
  if (item === null) {
    return ["previous", "up", "end"].includes(command)
      ? { kind: "series", seriesIndex: plan.series.length - 1, rawIndex: plan.rawCount - 1 }
      : { kind: "series", seriesIndex: 0, rawIndex: 0 };
  }
  if (command === "home") return { ...item, rawIndex: 0 };
  if (command === "end") return { ...item, rawIndex: plan.rawCount - 1 };
  if (command === "up" || command === "down") {
    const delta = command === "up" ? -1 : 1;
    return { ...item, seriesIndex: Math.min(Math.max(item.seriesIndex + delta, 0), plan.series.length - 1) };
  }
  const delta = command === "previous" ? -1 : 1;
  return { ...item, rawIndex: Math.min(Math.max(item.rawIndex + delta, 0), plan.rawCount - 1) };
}

export function nearestSeriesPoint(
  plan: PlotInspectionPlan,
  projected: readonly (readonly (readonly number[])[])[],
  pointerPx: readonly number[],
  hitRadiusPx = DEFAULT_PLOT_HIT_RADIUS_PX,
): PlotSelection | null {
  if (plan.kind !== "series") throw new RangeError("series picking requires a series plan");
  if (projected.length !== plan.series.length || pointerPx.length !== 2) {
    throw new RangeError("projected series must match the complete inspection plan");
  }
  const pointerX = finite(pointerPx[0], "pointer", true);
  const pointerY = finite(pointerPx[1], "pointer", true);
  const radius = finite(hitRadiusPx, "hit radius", true);
  if (radius <= 0 || radius > 100) throw new RangeError("hit radius must be a positive pixel distance");
  let nearest: readonly [number, number, number] | null = null;
  projected.forEach((points, seriesIndex) => {
    if (points.length !== plan.rawCount) throw new RangeError("projected series must match the complete inspection plan");
    points.forEach((point, rawIndex) => {
      if (point.length !== 2) throw new RangeError("projected point must contain two pixel coordinates");
      const distance = Math.hypot(
        finite(point[0], "projected point", true) - pointerX,
        finite(point[1], "projected point", true) - pointerY,
      );
      const candidate: readonly [number, number, number] = [distance, seriesIndex, rawIndex];
      if (nearest === null || candidate[0] < nearest[0] ||
          (candidate[0] === nearest[0] && (candidate[1] < nearest[1] ||
            (candidate[1] === nearest[1] && candidate[2] < nearest[2])))) nearest = candidate;
    });
  });
  return nearest !== null && nearest[0] <= radius
    ? { kind: "series", seriesIndex: nearest[1], rawIndex: nearest[2] }
    : null;
}

export function histogramBinAtData(
  plan: PlotInspectionPlan, xValue: unknown, yValue: unknown,
): PlotSelection | null {
  if (plan.kind !== "histogram") throw new RangeError("histogram picking requires a histogram plan");
  const x = finite(xValue, "histogram pointer");
  const y = finite(yValue, "histogram pointer");
  if (y < 0) return null;
  for (const item of plan.bins) {
    const inX = (item.lower <= x && x < item.upper) ||
      (item.index === plan.bins.length - 1 && x === item.upper);
    if (inX) return y <= item.count ? { kind: "histogram", binIndex: item.index } : null;
  }
  return null;
}
