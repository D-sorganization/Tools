/** Provenance-complete, presentation-only Morris selection tests. */

import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/morris_global_sensitivity_golden_v1.json";
import selectorFixture from "./__fixtures__/morris_selector_parity_v1.json";
import { parseMorrisReport } from "./morrisGlobalSensitivityContract";
import { presentMorrisReport } from "./morrisPresentation";
import {
  listMorrisSourceOptions,
  listMorrisTargetOptions,
  selectMorrisReport,
} from "./morrisSelectorContract";

const reportWithRepeatedName = () => {
  const document = structuredClone(fixture) as Record<string, unknown>;
  const estimates = document.estimates as Array<Record<string, unknown>>;
  const repeated = estimates
    .filter((estimate) => (estimate.target as Record<string, unknown>).name === "clubhead_x_m")
    .map((estimate) => {
      const copy = structuredClone(estimate);
      copy.target = {
        ...(copy.target as Record<string, unknown>),
        time_s: 0.04,
        point_id: "grip",
      };
      return copy;
    });
  document.estimates = [...estimates, ...repeated];
  return parseMorrisReport(document);
};

const parityReport = () => parseMorrisReport({
  schema_id: "swing-sim/morris-global-sensitivity-report",
  schema_version: 1,
  method: "morris-elementary-effects",
  design: { trajectories: 4, levels: 4, seed: 7, total_samples: 12, normalized_step: 2 / 3 },
  assumptions: ["model scenario"],
  interaction_caveat: "screening only",
  estimates: selectorFixture.targets.flatMap((target) => ["yaw", "forward"].map(
    (sourceId, sourceIndex) => ({
      source: {
        spec_id: sourceId, variable_key: `swing_sim.swing.${sourceId}_deg`, unit: "deg",
        bounds: [-1, 1], time_window_s: null, point_ids: [],
      },
      target,
      effects: { mu: 4 - sourceIndex, mu_star: 4 - sourceIndex, mu_star_standard_error: 0, sigma: 0 },
      availability: "available", sample_adequacy: "limited",
      denominator: {
        total_pairs: 4, valid_pairs: 4, typed_no_impact_pairs: 0,
        no_impact_unavailable_pairs: 0, failed_pairs: 0, nonfinite_pairs: 0,
      },
    }),
  )),
});

describe("Morris result selection", () => {
  it("keeps same-name point and phase observations distinct", () => {
    const report = reportWithRepeatedName();
    const repeated = listMorrisTargetOptions(report)
      .filter((option) => option.identity.name === "clubhead_x_m");

    expect(repeated.map((option) => [option.identity.pointId, option.identity.timeS]))
      .toEqual([["clubhead", 0.03], ["grip", 0.04]]);
    expect(repeated.every((option) => option.identity.schemaVersion === 1)).toBe(true);
  });

  it("matches the shared Python/TypeScript selector fixture", () => {
    const options = listMorrisTargetOptions(parityReport());
    expect(options.map((option) => option.label)).toEqual(selectorFixture.expected_labels);
    expect(options.filter((option) => option.identity.kind === "state-point")
      .map((option) => [option.identity.pointId, option.identity.timeS]))
      .toEqual(selectorFixture.expected_state_points);
  });

  it("filters one source while retaining its global rank and report identity", () => {
    const report = reportWithRepeatedName();
    const target = listMorrisTargetOptions(report)[0].identity;
    const sources = listMorrisSourceOptions(report, target);
    const all = selectMorrisReport(report, { target, sourceSpecId: null });
    const one = selectMorrisReport(report, { target, sourceSpecId: sources[1].specId });

    expect(one.rows).toHaveLength(1);
    expect(one.rows[0].rank).toBe(all.rows.find((row) => row.specId === sources[1].specId)?.rank);
    expect(Object.isFrozen(report)).toBe(true);
  });

  it("fails closed for a legacy ambiguous name", () => {
    expect(() => presentMorrisReport(reportWithRepeatedName(), "clubhead_x_m"))
      .toThrow(/ambiguous/);
  });
});
