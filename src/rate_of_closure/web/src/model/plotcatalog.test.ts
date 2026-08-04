/**
 * Catalog parity pins: the web key list must match the pytest-exported
 * fixture (`plotcatalog.fixture.json`, regenerated from
 * `rate_of_closure/plotting/catalog.py` and asserted by
 * tests/rate_of_closure/test_plotting.py on the Python side), and the
 * extractors must produce sane values from a reference run.
 */

import { describe, expect, it } from "vitest";

import fixture from "./plotcatalog.fixture.json";
import { DEFAULT_SCENARIO } from "./impact";
import {
  PLOT_CATALOG,
  catalogKeys,
  catalogVariable,
  isSeries,
  supportedByCategory,
  type PlotContext,
} from "./plotcatalog";
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

describe("plot catalog — parity with the Python catalog", () => {
  it("mirrors the pytest-exported key list exactly", () => {
    expect(fixture.format).toBe("rate_of_closure.plot_catalog/1");
    expect(catalogKeys()).toEqual(fixture.keys);
  });

  it("namespaces every key by its category", () => {
    const prefixes: Record<string, string> = {
      Input: "input.",
      "Swing Sample": "swing.",
      Impact: "impact.",
      Launch: "launch.",
      Flight: "flight.",
      Metric: "metric.",
    };
    for (const entry of PLOT_CATALOG) {
      expect(entry.key.startsWith(prefixes[entry.category]), entry.key).toBe(
        true,
      );
      expect(entry.label.length, entry.key).toBeGreaterThan(0);
    }
  });

  it("extracts finite scalars for every supported scalar entry", () => {
    const ctx = context();
    for (const entry of PLOT_CATALOG) {
      if (entry.extractor === null || isSeries(entry.key)) continue;
      const value = entry.extractor(ctx);
      expect(typeof value, entry.key).toBe("number");
      expect(Number.isFinite(value as number), entry.key).toBe(true);
    }
  });

  it("extracts matching-length series for every supported series entry", () => {
    const ctx = context();
    const lengths: Record<string, number> = {
      "Swing Sample": ctx.run.swing.length,
      Flight: ctx.run.flight.length,
    };
    for (const entry of PLOT_CATALOG) {
      if (entry.extractor === null || !isSeries(entry.key)) continue;
      const values = entry.extractor(ctx) as number[];
      expect(Array.isArray(values), entry.key).toBe(true);
      expect(values.length, entry.key).toBe(lengths[entry.category]);
    }
  });

  it("only hides the documented unsupported keys from the builder", () => {
    const unsupported = PLOT_CATALOG.filter((e) => e.extractor === null).map(
      (e) => e.key,
    );
    expect(unsupported).toEqual([
      "swing.angular_speed_dps",
      "impact.spin_loft_deg",
      "impact.spin_axis_tilt_deg",
      "impact.energy_transfer_j",
    ]);
    for (const category of ["Swing Sample", "Flight"] as const) {
      expect(supportedByCategory(category).length).toBeGreaterThan(0);
    }
  });

  it("rejects unknown keys", () => {
    expect(() => catalogVariable("nope.nothing")).toThrow(/unknown catalog/);
  });
});
