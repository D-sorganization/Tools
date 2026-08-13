import fixture from "./__fixtures__/variation_execution_document_v1.json";
import { describe, expect, it } from "vitest";

import type { VariationPlanTs } from "./variationSchema";
import {
  LEGACY_CURRENT_REGISTRY_WARNING,
  makeVariationExecutionMetadata,
  parseVariationExecutionDocument,
  resolveVariationExecutionMetadata,
  variationExecutionDocument,
} from "./variationExecutionMetadata";

const BALL_SPEED = "swing_sim.flight.launch.ball_speed_mph";
const LAUNCH_ANGLE = "swing_sim.flight.launch.launch_angle_deg";

type MutableExecutionDocument = ReturnType<typeof variationExecutionDocument>;
type ExecutionDocumentMutation = (value: MutableExecutionDocument) => void;

const plan = (seed = 17): VariationPlanTs => ({
  mode: "launch",
  baseVariables: { [BALL_SPEED]: 154.25 },
  noise: [{
    variableKey: LAUNCH_ANGLE,
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
  seed,
  flightModel: "waterloo_penner",
});

describe("variation execution metadata", () => {
  it("snapshots resolved values, units, dimensions, and immutable identities", () => {
    const metadata = makeVariationExecutionMetadata(plan());
    const snapshots = new Map(metadata.resolvedVariables.map((item) => [item.variableKey, item]));

    expect(metadata).toMatchObject({
      schemaId: "rate-of-closure/variation-execution-metadata",
      schemaVersion: 1,
      registrySchemaVersion: 1,
      mode: "launch",
      flightModel: "waterloo_penner",
    });
    expect(metadata.planSha256).toMatch(/^[0-9a-f]{64}$/);
    expect(metadata.registrySha256).toMatch(/^[0-9a-f]{64}$/);
    expect(snapshots.get(BALL_SPEED)).toMatchObject({ value: 154.25, unit: "mph", dimension: "speed" });
    expect(snapshots.get(LAUNCH_ANGLE)).toMatchObject({ value: 12, unit: "deg", dimension: "angle" });
    expect(Object.isFrozen(metadata)).toBe(true);
    expect(Object.isFrozen(metadata.resolvedVariables)).toBe(true);
  });

  it("matches and strictly reads the shared Python fixture", () => {
    expect(variationExecutionDocument(plan())).toEqual(fixture);
    expect(parseVariationExecutionDocument(JSON.stringify(fixture))).toEqual({
      plan: plan(),
      metadata: makeVariationExecutionMetadata(plan()),
      warning: null,
    });
  });

  it("round-trips the React ball-setup extension without changing plan-v2", () => {
    const teePlan: VariationPlanTs = {
      ...plan(),
      ballSetup: { supportMode: "tee", teeHeightM: 0.0381 },
    };
    const document = variationExecutionDocument(teePlan);

    expect(document.plan.ball_setup).toEqual({
      support_mode: "tee",
      tee_height_m: 0.0381,
      height_reference: "ground_plane_to_ball_bottom",
      ball_center_m: [0, 0.059435, 0],
    });
    expect(parseVariationExecutionDocument(JSON.stringify(document)).plan).toEqual(teePlan);
    expect(document.plan).not.toHaveProperty("execution_metadata");
  });

  it("accepts semantically canonical plan fields independent of JSON key order", () => {
    const document = variationExecutionDocument(plan());
    document.plan = Object.fromEntries(Object.entries(document.plan).reverse());

    expect(parseVariationExecutionDocument(JSON.stringify(document)).plan).toEqual(plan());
  });

  const driftCases: Array<[string, ExecutionDocumentMutation]> = [
    ["plan digest", (value) => { value.plan.seed = 18; }],
    ["flight model", (value) => { value.metadata.flight_model = "nathan"; }],
    ["resolved variable snapshot", (value) => { value.metadata.resolved_variables[0].value = 999; }],
    ["resolved variable snapshot", (value) => { value.metadata.resolved_variables[0].unit = "m/s"; }],
    ["resolved variable snapshot", (value) => { value.metadata.resolved_variables[0].dimension = "length"; }],
    ["resolved variable snapshot", (value) => { value.metadata.resolved_variables.pop(); }],
    ["resolved variable snapshot", (value) => {
      value.metadata.resolved_variables.push({ ...value.metadata.resolved_variables[0] });
    }],
    ["registry digest", (value) => { value.metadata.registry_sha256 = "0".repeat(64); }],
    ["metadata fields", (value) => { Object.assign(value.metadata, { unexpected: true }); }],
    ["execution document fields", (value) => { Object.assign(value, { unexpected: true }); }],
  ];

  it.each(driftCases)("rejects %s drift", (message, mutate) => {
    const document = structuredClone(variationExecutionDocument(plan()));
    mutate(document);

    expect(() => parseVariationExecutionDocument(JSON.stringify(document))).toThrow(
      new RegExp(message, "i"),
    );
  });

  it("rejects cross-plan metadata and explicitly warns for legacy plans", () => {
    const metadata = makeVariationExecutionMetadata(plan(17));
    expect(() => resolveVariationExecutionMetadata(plan(18), metadata)).toThrow(/plan digest/i);

    expect(resolveVariationExecutionMetadata(plan(), null)).toEqual({
      metadata: makeVariationExecutionMetadata(plan()),
      warning: LEGACY_CURRENT_REGISTRY_WARNING,
    });
  });
});
