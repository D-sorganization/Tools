import reactV2Fixture from "./__fixtures__/variation_execution_document_react_v2.json";
import { describe, expect, it } from "vitest";

import { planToJson, type VariationPlanTs } from "./variationSchema";
import {
  parsePersistedVariationPlanBinding,
  persistedVariationPlanBinding,
  persistedVariationPlanJson,
  parsePersistedVariationPlan,
} from "./variationPersistedPlan";

const plan: VariationPlanTs = {
  mode: "launch",
  baseVariables: { "swing_sim.flight.launch.ball_speed_mph": 154.25 },
  noise: [{
    variableKey: "swing_sim.flight.launch.launch_angle_deg",
    distribution: "normal",
    scale: 0.75,
    lower: null,
    upper: null,
    specId: "launch-angle",
    timeWindowS: null,
    pointIds: [],
  }],
  groups: [],
  nRuns: 8,
  seed: 17,
  flightModel: "waterloo_penner",
};

describe("persisted variation plans", () => {
  it("round-trips a canonical document without warning", () => {
    const resolution = parsePersistedVariationPlan(persistedVariationPlanJson(plan));

    expect(resolution.plan).toEqual(plan);
    expect(resolution.metadata).not.toBeNull();
    expect(resolution.provenance).not.toBeNull();
    expect(resolution.warning).toBeNull();
  });

  it("loads raw v2 only with a current-registry warning", () => {
    const resolution = parsePersistedVariationPlan(planToJson(plan));

    expect(resolution.plan).toEqual(plan);
    expect(resolution.metadata).toBeNull();
    expect(resolution.provenance).toBeNull();
    expect(resolution.warning).toMatch(/not evidence of historical reproducibility/i);
  });

  it("loads a v2 execution document without inventing source provenance", () => {
    const resolution = parsePersistedVariationPlan(JSON.stringify(reactV2Fixture));

    expect(resolution.metadata).toBeNull();
    expect(resolution.provenance).toBeNull();
    expect(resolution.warning).toMatch(/source provenance was not recorded/i);
  });

  it("rejects duplicate fields before migration", () => {
    const duplicate = planToJson(plan).replace('"seed": 17', '"seed": 17, "seed": 18');

    expect(() => parsePersistedVariationPlan(duplicate)).toThrow(/duplicate JSON field: seed/);
  });

  it("round-trips canonical and legacy bindings without inventing evidence", () => {
    const canonical = parsePersistedVariationPlan(persistedVariationPlanJson(plan));
    const legacy = parsePersistedVariationPlan(planToJson(plan));

    expect(parsePersistedVariationPlanBinding(
      persistedVariationPlanBinding(canonical),
    )).toEqual(canonical);
    expect(parsePersistedVariationPlanBinding(
      persistedVariationPlanBinding(legacy),
    )).toEqual(legacy);
  });

  it("rejects substituted binding evidence", () => {
    const binding = persistedVariationPlanBinding(plan) as Record<string, unknown>;
    const document = binding.document as Record<string, unknown>;
    const boundPlan = document.plan as Record<string, unknown>;
    boundPlan.seed = 99;

    expect(() => parsePersistedVariationPlanBinding(binding)).toThrow(/digest/i);
  });
});
