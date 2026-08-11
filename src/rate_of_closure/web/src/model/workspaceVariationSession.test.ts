import fixture from "./__fixtures__/workspace_variation_parity.json";
import { describe, expect, it } from "vitest";

import { outputsForMode, planFromJson } from "./variation";
import {
  migratedLegacyVariationFallback,
  variationWorkspaceDocument,
  variationWorkspaceFromDocument,
} from "./workspaceVariationSession";

const plan = () => planFromJson(JSON.stringify(fixture.plan));

describe("variation workspace selection parity", () => {
  it("exposes the complete canonical swing-output contract", () => {
    expect(outputsForMode("swing")).toEqual([
      "candidate_time_s",
      "closest_approach_m",
      "contact_margin_m",
      "impact_time_s",
      "clubhead_speed_mps",
      "spin_loft_deg",
      "face_to_path_deg",
      "spin_axis_tilt_deg",
      "ball_speed_mph",
      "launch_angle_deg",
      "launch_azimuth_deg",
      "spin_rpm",
      "carry_m",
      "lateral_m",
      "max_height_m",
      "flight_time_s",
      "landing_angle_deg",
    ]);
  });

  it("round trips the authored specification without storing results", () => {
    const state = variationWorkspaceFromDocument(fixture.selection, plan());

    expect(state.analysisExecution).toBe("both");
    expect(state.selectedOutputMetrics).toEqual([
      "carry_m",
      "lateral_m",
      "apex_m",
    ]);
    expect(state.plan.nRuns).toBe(300);
    expect(state.plan.seed).toBe(42);
    expect(variationWorkspaceDocument(state)).toEqual(fixture.selection);
  });

  it.each([
    ["analysis_execution", "parallel", /execution/i],
    ["selected_output_metrics", ["unknown_metric"], /metric/i],
    ["selected_output_metrics", ["carry_m", "carry_m"], /unique/i],
  ])("rejects invalid %s", (field, value, error) => {
    const selection = structuredClone(fixture.selection) as Record<
      string,
      unknown
    >;
    (selection.data as Record<string, unknown>)[field] = value;

    expect(() => variationWorkspaceFromDocument(selection, plan())).toThrow(
      error,
    );
  });

  it("fails closed when a legacy root plan conflicts with the live fallback", () => {
    const state = variationWorkspaceFromDocument(fixture.selection, plan());
    expect(migratedLegacyVariationFallback(state, null)).toEqual(state);
    expect(migratedLegacyVariationFallback(state, plan())).toEqual(state);

    const conflict = { ...plan(), seed: plan().seed + 1 };
    expect(() => migratedLegacyVariationFallback(state, conflict)).toThrow(
      /conflicts/i,
    );
  });
});
