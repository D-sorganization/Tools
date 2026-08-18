/**
 * PlotSpec JSON round-trip + compute-pipeline tests for the web port,
 * mirroring tests/rate_of_closure/test_plotting.py.
 */

import { describe, expect, it } from "vitest";

import { DEFAULT_SCENARIO } from "./impact";
import { type PlotContext } from "./plotcatalog";
import {
  BUILTIN_PLOTS,
  computePlotData,
  plotDataCsv,
  plotDataJson,
  specFromJson,
  specToJson,
  type PlotSpec,
} from "./plotspec";
import { runSimulation, type SimulationInput } from "./simulation";

const context = (): PlotContext => {
  const input: SimulationInput = {
    sourceKind: "manual",
    clubheadSpeedMph: DEFAULT_SCENARIO.clubheadSpeedMph,
    omegaDps: [0, 0, 0],
    loftDeg: 10.5,
    impactOffsetToeMm: 0,
    impactOffsetHighMm: 0,
    planeYawDeg: 0,
    planeSideTiltDeg: -45,
    planeForwardTiltDeg: 0,
    impactTimeS: null,
    swingDurationS: 1.5,
  };
  return { scenario: DEFAULT_SCENARIO, input, run: runSimulation(input) };
};

const fast = (spec: PlotSpec): PlotSpec =>
  spec.kind === "sweep" ? { ...spec, x_count: 4 } : spec;

describe("plot spec — JSON round-trip", () => {
  it("preserves every field through the shared schema", () => {
    const spec: PlotSpec = {
      kind: "sweep",
      x_key: "input.omega_shaft_dps",
      y_keys: ["metric.path_deviation_deg", "launch.spin_rpm"],
      series_key: null,
      title: "Round Trip",
      x_log: false,
      y_log: true,
      x_start: 0,
      x_stop: 4000,
      x_count: 11,
    };
    expect(specFromJson(specToJson(spec))).toEqual(spec);
  });

  it("rejects foreign formats, kinds, and keys", () => {
    expect(() => specFromJson({ format: "other/9" })).toThrow(/format/);
    expect(() =>
      specFromJson({
        format: "rate_of_closure.plot_spec/1",
        kind: "pie",
        x_key: "swing.time_s",
        y_keys: ["swing.speed_mps"],
      }),
    ).toThrow(/unknown plot kind/);
    expect(() =>
      specFromJson({
        format: "rate_of_closure.plot_spec/1",
        kind: "line",
        x_key: "bogus.key",
        y_keys: ["swing.speed_mps"],
      }),
    ).toThrow(/unknown catalog/);
    expect(() => specFromJson({
      format: "rate_of_closure.plot_spec/1", kind: "histogram",
      x_key: "flight.speed_mps", y_keys: [], y_log: true,
    })).toThrow(/count axis/);
  });

  it("accepts a definition exported by the desktop app", () => {
    // Literal payload shape produced by Python spec_to_json().
    const payload = {
      format: "rate_of_closure.plot_spec/1",
      kind: "line",
      x_key: "flight.x_m",
      y_keys: ["flight.y_m"],
      series_key: null,
      title: "Flight Profile — Height vs Downrange Distance",
      x_log: false,
      y_log: false,
      x_start: null,
      x_stop: null,
      x_count: 25,
    };
    const spec = specFromJson(payload);
    expect(spec.x_key).toBe("flight.x_m");
    expect(specToJson(spec)).toEqual(payload);
  });
});

describe("compute pipeline", () => {
  const ctx = context();

  it("renders every web builtin without error", () => {
    for (const entry of BUILTIN_PLOTS) {
      const data = computePlotData(fast(entry.make(0.06)), ctx);
      expect(data.x.length, entry.name).toBeGreaterThanOrEqual(2);
      for (const series of data.series)
        expect(series.values.length, entry.name).toBe(data.x.length);
      expect(data.xLabel.length, entry.name).toBeGreaterThan(0);
    }
  });

  it("sweeps re-run the simulation per grid point", () => {
    const spec = fast(
      BUILTIN_PLOTS.find((b) => b.name === "launch_vs_toe_offset")!.make(0.06),
    );
    const data = computePlotData(spec, ctx);
    const speeds = data.series.find((s) => s.label === "Ball Speed")!.values;
    // Center strike must be at least as fast as a 20 mm toe strike.
    expect(Math.max(...speeds)).toBeGreaterThan(speeds[0]);
  });

  it("exports well-formed CSV and JSON of the plotted data", () => {
    const spec = BUILTIN_PLOTS.find(
      (b) => b.name === "flight_profile_side",
    )!.make(0.06);
    const data = computePlotData(spec, ctx);
    const csv = plotDataCsv(data);
    const lines = csv.trim().split("\n");
    expect(lines[0]).toBe("Downrange Distance [m],Height");
    expect(lines.length).toBe(data.x.length + 1);
    const payload = JSON.parse(plotDataJson(data));
    expect(payload.format).toBe("rate_of_closure.plot_data/1");
    expect(payload.rows.length).toBe(data.x.length);
    expect(specFromJson(payload.spec)).toEqual(spec);
  });

  it("publishes a deeply frozen PlotData snapshot", () => {
    const spec = BUILTIN_PLOTS.find((b) => b.name === "swing_time_series")!.make(0.06);
    const data = computePlotData(spec, ctx);
    expect(Object.isFrozen(data)).toBe(true);
    expect(Object.isFrozen(data.spec)).toBe(true);
    expect(Object.isFrozen(data.spec.y_keys)).toBe(true);
    expect(Object.isFrozen(data.x)).toBe(true);
    expect(Object.isFrozen(data.series[0].values)).toBe(true);
    expect(() => ((data.x as number[])[0] = 99)).toThrow();
    expect(() => ((data.series[0].values as number[])[0] = 99)).toThrow();
    expect(() => ((data.spec.y_keys as string[])[0] = "metric.carry_m")).toThrow();
  });
});
