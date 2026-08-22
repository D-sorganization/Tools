/**
 * Plot definitions + the compute pipeline for the web plotting suite.
 *
 * Same JSON schema as the Python `plotting/spec.py`
 * (`rate_of_closure.plot_spec/1`), so definitions exported from either
 * UI import into the other. `computePlotData` mirrors
 * `plotting/render.py`: line / scatter / histogram read per-sample
 * series off one run; `sweep` re-runs the TS simulation across the
 * swept input's grid and extracts scalar outputs per point.
 */

import {
  axisLabel,
  catalogVariable,
  isSeries,
  type PlotContext,
} from "./plotcatalog";
import { type ImpactScenario } from "./impact";
import { snapshotPlotData } from "./plotDataSnapshot";
import {
  runSimulation,
  type SimulationInput,
  type SimulationRunTs,
} from "./simulation";

export const SPEC_FORMAT = "rate_of_closure.plot_spec/1";

export type PlotKind = "line" | "scatter" | "sweep" | "histogram";

export interface PlotSpec {
  readonly kind: PlotKind;
  readonly x_key: string;
  readonly y_keys: readonly string[];
  readonly series_key: string | null;
  readonly title: string;
  readonly x_log: boolean;
  readonly y_log: boolean;
  readonly x_start: number | null;
  readonly x_stop: number | null;
  readonly x_count: number;
}

export interface PlotData {
  spec: PlotSpec;
  x: readonly number[];
  /** Series label -> values (same length as x). Empty for histograms. */
  series: readonly { readonly label: string; readonly values: readonly number[] }[];
  xLabel: string;
  yLabel: string;
}

export type PlotSimulationExecutor = (input: SimulationInput) => SimulationRunTs;

const SWEEP_Y_CATEGORIES = new Set(["Impact", "Launch", "Metric"]);

/** Validate a definition; throws with a DbC-style message when invalid. */
export function validateSpec(spec: PlotSpec): void {
  if (!["line", "scatter", "sweep", "histogram"].includes(spec.kind))
    throw new Error(`unknown plot kind ${spec.kind}`);
  catalogVariable(spec.x_key);
  for (const key of spec.y_keys) catalogVariable(key);
  if (spec.series_key !== null) catalogVariable(spec.series_key);
  if (spec.kind === "histogram") {
    if (!isSeries(spec.x_key))
      throw new Error("histogram needs a per-sample x variable");
    if (spec.y_log) throw new Error("histogram count axis cannot be logarithmic");
    return;
  }
  if (spec.kind === "sweep") {
    if (catalogVariable(spec.x_key).category !== "Input")
      throw new Error("sweep x_key must be an Input variable");
    if (spec.y_keys.length === 0) throw new Error("sweep needs y_keys");
    for (const key of spec.y_keys)
      if (!SWEEP_Y_CATEGORIES.has(catalogVariable(key).category))
        throw new Error("sweep y_keys must be scalar outputs");
    if (spec.x_start === null || spec.x_stop === null)
      throw new Error("sweep needs x_start and x_stop");
    if (!(spec.x_start < spec.x_stop))
      throw new Error("x_start must be < x_stop");
    if (spec.x_count < 2 || spec.x_count > 501)
      throw new Error("x_count must be in [2, 501]");
    return;
  }
  if (spec.y_keys.length === 0) throw new Error(`${spec.kind} needs y_keys`);
  if (!isSeries(spec.x_key))
    throw new Error(`${spec.kind} x_key must be a per-sample variable`);
  for (const key of spec.y_keys)
    if (!isSeries(key))
      throw new Error(`${spec.kind} y_keys must be per-sample variables`);
}

/** Definition -> the shared JSON schema. */
export function specToJson(spec: PlotSpec): Record<string, unknown> {
  validateSpec(spec);
  return { format: SPEC_FORMAT, ...spec };
}

/** Shared JSON schema -> validated definition. */
export function specFromJson(data: unknown): PlotSpec {
  if (typeof data !== "object" || data === null)
    throw new Error("plot definition must be an object");
  const record = data as Record<string, unknown>;
  if (record.format !== SPEC_FORMAT)
    throw new Error(`unsupported plot definition format ${String(record.format)}`);
  const spec: PlotSpec = {
    kind: record.kind as PlotKind,
    x_key: String(record.x_key),
    y_keys: Array.isArray(record.y_keys) ? record.y_keys.map(String) : [],
    series_key:
      record.series_key === undefined || record.series_key === null
        ? null
        : String(record.series_key),
    title: typeof record.title === "string" ? record.title : "",
    x_log: Boolean(record.x_log),
    y_log: Boolean(record.y_log),
    x_start: typeof record.x_start === "number" ? record.x_start : null,
    x_stop: typeof record.x_stop === "number" ? record.x_stop : null,
    x_count: typeof record.x_count === "number" ? record.x_count : 25,
  };
  validateSpec(spec);
  return spec;
}

/** Sweepable input key -> how it lands in a new scenario/input pair. */
function withInputValue(
  scenario: ImpactScenario,
  input: SimulationInput,
  key: string,
  value: number,
): { scenario: ImpactScenario; input: SimulationInput } {
  const scenarioFields: Record<string, keyof ImpactScenario> = {
    "input.clubhead_speed_mph": "clubheadSpeedMph",
    "input.omega_plane_dps": "omegaPlaneDps",
    "input.omega_shaft_dps": "omegaShaftDps",
    "input.lie_angle_deg": "lieAngleDeg",
    "input.com_to_face_mm": "comToFaceMm",
    "input.impact_offset_toe_mm": "impactOffsetToeMm",
    "input.impact_offset_high_mm": "impactOffsetHighMm",
    "input.contact_duration_us": "contactDurationUs",
  };
  const inputFields: Record<string, keyof SimulationInput> = {
    "input.plane_yaw_deg": "planeYawDeg",
    "input.plane_side_tilt_deg": "planeSideTiltDeg",
    "input.plane_forward_tilt_deg": "planeForwardTiltDeg",
    "input.impact_time_s": "impactTimeS",
  };
  if (key in scenarioFields) {
    const next = { ...scenario, [scenarioFields[key]]: value };
    const nextInput = { ...input };
    if (key === "input.clubhead_speed_mph") nextInput.clubheadSpeedMph = value;
    if (key === "input.impact_offset_toe_mm")
      nextInput.impactOffsetToeMm = value;
    if (key === "input.impact_offset_high_mm")
      nextInput.impactOffsetHighMm = value;
    return { scenario: next, input: nextInput };
  }
  if (key in inputFields)
    return { scenario, input: { ...input, [inputFields[key]]: value } };
  throw new Error(`cannot sweep ${key}`);
}

function extractScalar(ctx: PlotContext, key: string): number {
  const extractor = catalogVariable(key).extractor;
  if (!extractor) throw new Error(`${key} is not extractable on the web`);
  const value = extractor(ctx);
  if (Array.isArray(value)) throw new Error(`${key} is not a scalar`);
  return value;
}

function extractArray(ctx: PlotContext, key: string): number[] {
  const extractor = catalogVariable(key).extractor;
  if (!extractor) throw new Error(`${key} is not extractable on the web`);
  const value = extractor(ctx);
  if (!Array.isArray(value)) throw new Error(`${key} is not a series`);
  return value;
}

function sharedYLabel(yKeys: readonly string[]): string {
  if (yKeys.length === 1) return axisLabel(yKeys[0]);
  const units = new Set(yKeys.map((key) => catalogVariable(key).unit));
  if (units.size === 1) {
    const [unit] = units;
    return unit ? `Value [${unit}]` : "Value";
  }
  return "Value (Mixed Units)";
}

function sweepData(
  spec: PlotSpec,
  ctx: PlotContext,
  executeSimulation: PlotSimulationExecutor,
): PlotData {
  const start = spec.x_start as number;
  const stop = spec.x_stop as number;
  const xs: number[] = [];
  const columns = spec.y_keys.map(() => [] as number[]);
  for (let i = 0; i < spec.x_count; i += 1) {
    const value = start + ((stop - start) * i) / (spec.x_count - 1);
    try {
      const swept = withInputValue(ctx.scenario, ctx.input, spec.x_key, value);
      const run = executeSimulation(swept.input);
      const pointCtx: PlotContext = { ...swept, run };
      const row = spec.y_keys.map((key) => extractScalar(pointCtx, key));
      xs.push(value);
      row.forEach((y, j) => columns[j].push(y));
    } catch {
      // Skip infeasible sweep points, matching the Python pipeline.
    }
  }
  if (xs.length < 2)
    throw new Error("sweep produced fewer than 2 feasible points");
  return snapshotPlotData({
    spec,
    x: xs,
    series: spec.y_keys.map((key, j) => ({
      label: catalogVariable(key).label,
      values: columns[j],
    })),
    xLabel: axisLabel(spec.x_key),
    yLabel: sharedYLabel(spec.y_keys),
  });
}

/** Evaluate a definition against the current scenario / run context. */
export function computePlotData(
  spec: PlotSpec,
  ctx: PlotContext,
  executeSimulation: PlotSimulationExecutor = runSimulation,
): PlotData {
  validateSpec(spec);
  if (spec.kind === "sweep") return sweepData(spec, ctx, executeSimulation);
  const x = extractArray(ctx, spec.x_key);
  if (spec.kind === "histogram")
    return snapshotPlotData({
      spec,
      x,
      series: [],
      xLabel: axisLabel(spec.x_key),
      yLabel: "Count",
    });
  return snapshotPlotData({
    spec,
    x,
    series: spec.y_keys.map((key) => ({
      label: catalogVariable(key).label,
      values: extractArray(ctx, key),
    })),
    xLabel: axisLabel(spec.x_key),
    yLabel: sharedYLabel(spec.y_keys),
  });
}

const spec = (partial: Partial<PlotSpec> & Pick<PlotSpec, "kind" | "x_key">): PlotSpec => ({
  y_keys: [],
  series_key: null,
  title: "",
  x_log: false,
  y_log: false,
  x_start: null,
  x_stop: null,
  x_count: 25,
  ...partial,
});

/** Built-in plots the web port supports, mirroring the desktop set. */
export const BUILTIN_PLOTS: Array<{ name: string; label: string; make: (swingDurationS: number) => PlotSpec }> = [
  {
    name: "closure_sweep",
    label: "Closure Sweep",
    make: () =>
      spec({
        kind: "sweep",
        x_key: "input.omega_shaft_dps",
        y_keys: ["metric.path_deviation_deg"],
        title: "Impact-Point Path Deviation vs About-Shaft Rotation Rate",
        x_start: 0,
        x_stop: 4000,
        x_count: 41,
      }),
  },
  {
    name: "delivery_vs_tau",
    label: "Delivery vs τ Sweep",
    make: (duration) =>
      spec({
        kind: "sweep",
        x_key: "input.impact_time_s",
        y_keys: [
          "impact.club_path_deg",
          "impact.attack_angle_deg",
          "impact.face_to_path_deg",
        ],
        title: "Delivery vs Impact-Time Offset (τ)",
        x_start: 0.1 * duration,
        x_stop: 0.9 * duration,
        x_count: 21,
      }),
  },
  {
    name: "launch_vs_toe_offset",
    label: "Launch vs Toe Offset",
    make: () =>
      spec({
        kind: "sweep",
        x_key: "input.impact_offset_toe_mm",
        y_keys: ["launch.ball_speed_mph", "launch.spin_rpm"],
        title: "Launch vs Toe Impact Offset",
        x_start: -20,
        x_stop: 20,
        x_count: 21,
      }),
  },
  {
    name: "launch_vs_high_offset",
    label: "Launch vs High Offset",
    make: () =>
      spec({
        kind: "sweep",
        x_key: "input.impact_offset_high_mm",
        y_keys: ["launch.ball_speed_mph", "launch.spin_rpm"],
        title: "Launch vs Vertical Impact Offset",
        x_start: -10,
        x_stop: 10,
        x_count: 21,
      }),
  },
  {
    name: "swing_time_series",
    label: "Swing Time Series",
    make: () =>
      spec({
        kind: "line",
        x_key: "swing.time_s",
        y_keys: ["swing.speed_mps"],
        title: "Swing Time Series (Clubhead Speed)",
      }),
  },
  {
    name: "flight_profile_side",
    label: "Flight Profile (Side)",
    make: () =>
      spec({
        kind: "line",
        x_key: "flight.x_m",
        y_keys: ["flight.y_m"],
        title: "Flight Profile — Height vs Downrange Distance",
      }),
  },
  {
    name: "flight_profile_top",
    label: "Flight Profile (Top-Down)",
    make: () =>
      spec({
        kind: "line",
        x_key: "flight.x_m",
        y_keys: ["flight.z_m"],
        title: "Flight Profile — Top-Down Lateral vs Downrange Distance",
      }),
  },
];

/** CSV of the plotted numbers (header + rows), matching the desktop export. */
export function plotDataCsv(data: PlotData): string {
  // ⚡ Bolt Optimization: Replace chained array .map().join() with a single-pass loop
  let csv = data.xLabel;
  for (let i = 0; i < data.series.length; i++) {
    csv += "," + data.series[i].label;
  }
  csv += "\n";
  for (let i = 0; i < data.x.length; i += 1) {
    csv += data.x[i];
    for (let j = 0; j < data.series.length; j++) {
      csv += "," + data.series[j].values[i];
    }
    csv += "\n";
  }
  return csv;
}

/** JSON payload of the plotted numbers + definition, matching the desktop. */
export function plotDataJson(data: PlotData): string {
  return JSON.stringify(
    {
      format: "rate_of_closure.plot_data/1",
      spec: specToJson(data.spec),
      columns: [data.xLabel, ...data.series.map((s) => s.label)],
      rows: data.x.map((x, i) => [
        x,
        ...data.series.map((s) => s.values[i]),
      ]),
    },
    null,
    2,
  );
}
