import fixture from "./__fixtures__/variation_execution_document_v1.json";
import edgeFixture from "./__fixtures__/variation_execution_document_edge_floats_v1.json";
import pythonV2Fixture from "./__fixtures__/variation_execution_document_python_v2.json";
import reactV2Fixture from "./__fixtures__/variation_execution_document_react_v2.json";
import pythonV3Fixture from "./__fixtures__/variation_execution_document_python_v3.json";
import reactV3Fixture from "./__fixtures__/variation_execution_document_react_v3.json";
import { describe, expect, it } from "vitest";

import type { VariationPlanTs } from "./variationSchema";
import {
  LEGACY_CURRENT_REGISTRY_WARNING,
  LEGACY_EXECUTION_DOCUMENT_MIGRATION_ERROR,
  makePlanProducerProvenance,
  makeVariationExecutionMetadata,
  parseVariationExecutionDocument,
  REACT_DEFAULT_PROVENANCE,
  resolveVariationExecutionMetadata,
  variationExecutionDocument,
  variationExecutionDocumentJson,
} from "./variationExecutionMetadata";

const BALL_SPEED = "swing_sim.flight.launch.ball_speed_mph";
const LAUNCH_ANGLE = "swing_sim.flight.launch.launch_angle_deg";
const LAUNCH_AZIMUTH = "swing_sim.flight.launch.launch_azimuth_deg";

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

const edgePlan = (): VariationPlanTs => ({
  ...plan(Number.MAX_SAFE_INTEGER),
  baseVariables: {
    [BALL_SPEED]: 154.00000000000003,
    [LAUNCH_AZIMUTH]: -0,
    "swing_sim.flight.launch.spin_axis_deg": 1.0000000000000002,
  },
  noise: [{
    variableKey: LAUNCH_ANGLE,
    distribution: "normal",
    scale: 0.5000000000000001,
    lower: null,
    upper: null,
    specId: "edge-angle",
    timeWindowS: null,
    pointIds: [],
  }],
});

describe("variation execution metadata", () => {
  it("snapshots resolved values, units, dimensions, and immutable identities", () => {
    const metadata = makeVariationExecutionMetadata(plan());
    const snapshots = new Map(metadata.resolvedVariables.map((item) => [item.variableKey, item]));

    expect(metadata).toMatchObject({
      schemaId: "rate-of-closure/variation-execution-metadata",
      schemaVersion: 3,
      registrySchemaVersion: 1,
      mode: "launch",
      flightModel: "waterloo_penner",
    });
    expect(metadata.planSha256).toMatch(/^[0-9a-f]{64}$/);
    expect(metadata.registrySha256).toMatch(/^[0-9a-f]{64}$/);
    expect(metadata.rngIdentity.algorithmId).toBe("mulberry32-u32");
    expect(metadata.implementationIdentity.runtimeId).toBe("rate-of-closure/react");
    expect(metadata.implementationIdentity.solverId).toContain("waterloo-fixed-rk4-flight");
    expect(metadata.provenanceSha256).toMatch(/^[0-9a-f]{64}$/);
    expect(snapshots.get(BALL_SPEED)).toMatchObject({ value: 154.25, unit: "mph", dimension: "speed" });
    expect(snapshots.get(LAUNCH_ANGLE)).toMatchObject({ value: 12, unit: "deg", dimension: "angle" });
    expect(Object.isFrozen(metadata)).toBe(true);
    expect(Object.isFrozen(metadata.resolvedVariables)).toBe(true);
  });

  it("fails closed on v1 and v2 with an explicit migration instruction", () => {
    expect(() => parseVariationExecutionDocument(JSON.stringify(fixture))).toThrow(
      /Historical replay remains unproven/,
    );
    expect(() => parseVariationExecutionDocument(JSON.stringify(edgeFixture))).toThrow(/@1/);
    expect(() => parseVariationExecutionDocument(JSON.stringify(reactV2Fixture))).toThrow(
      /source provenance/,
    );
    expect(LEGACY_EXECUTION_DOCUMENT_MIGRATION_ERROR).toContain("fresh @3 document");
  });

  it("preserves signed-zero and edge-float digest under @3", () => {
    expect(parseVariationExecutionDocument(
      JSON.stringify(variationExecutionDocument(edgePlan())),
    ).metadata.planSha256).toBe(
      "6d7c23bb72a53359faa36d1d57d95835c9808bcdb67e0919859893e1a0cd711a",
    );
  });

  it("preserves the shared v2 plan and registry digests while requiring v3 provenance", () => {
    const document = variationExecutionDocument(plan());
    expect(reactV2Fixture.metadata.plan_sha256).toBe(pythonV2Fixture.metadata.plan_sha256);
    expect(reactV2Fixture.metadata.registry_sha256).toBe(pythonV2Fixture.metadata.registry_sha256);
    expect(document.metadata.plan_sha256).toBe(reactV2Fixture.metadata.plan_sha256);
    expect(document.metadata.registry_sha256).toBe(reactV2Fixture.metadata.registry_sha256);
    expect(parseVariationExecutionDocument(JSON.stringify(document)).provenance).toEqual(
      REACT_DEFAULT_PROVENANCE,
    );
    expect(document.schema_version).toBe(3);
    expect(() => parseVariationExecutionDocument(JSON.stringify(pythonV2Fixture))).toThrow(
      /source provenance/i,
    );
  });

  it("pins the React v3 golden and rejects the distinct Python runtime identity", () => {
    expect(variationExecutionDocument(plan())).toEqual(reactV3Fixture);
    expect(reactV3Fixture.metadata.plan_sha256).toBe(pythonV3Fixture.metadata.plan_sha256);
    expect(reactV3Fixture.metadata.registry_sha256).toBe(
      pythonV3Fixture.metadata.registry_sha256,
    );
    expect(parseVariationExecutionDocument(JSON.stringify(reactV3Fixture)).plan).toEqual(plan());
    expect(() => parseVariationExecutionDocument(JSON.stringify(pythonV3Fixture))).toThrow(
      /RNG identity/i,
    );
  });

  it("canonically serializes independent of input field order", () => {
    const document = variationExecutionDocument(plan());
    const reordered = Object.fromEntries(Object.entries(document).reverse());

    expect(variationExecutionDocumentJson(plan())).toBe(
      variationExecutionDocumentJson(parseVariationExecutionDocument(JSON.stringify(reordered))),
    );
    expect(variationExecutionDocumentJson(plan())).not.toContain("\n");
    expect(variationExecutionDocumentJson(plan())).toMatch(/^\{"metadata":/);
  });

  it("binds exact or explicitly unavailable source provenance", () => {
    const exact = makePlanProducerProvenance({
      producerId: "rate-of-closure/react",
      producerVersion: 1,
      sourceRepository: "D-sorganization/Tools",
      sourceRevision: "6e7f464d04bd7727ba4b985a399114ece8f0f723",
      sourceRevisionStatus: "exact",
      sourceRevisionReason: null,
    });
    const document = variationExecutionDocument(plan(), undefined, exact);

    expect(parseVariationExecutionDocument(JSON.stringify(document)).provenance).toEqual(exact);
    expect(() => makePlanProducerProvenance({
      ...exact, sourceRevision: "development",
    })).toThrow(/exact source revision/i);
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

  it("normalizes signed zero in the document, snapshot, and plan digest", () => {
    const negative = {
      ...plan(),
      baseVariables: { ...plan().baseVariables, [LAUNCH_AZIMUTH]: -0 },
    };
    const positive = {
      ...negative,
      baseVariables: { ...negative.baseVariables, [LAUNCH_AZIMUTH]: 0 },
    };
    const document = variationExecutionDocument(negative);
    const baseVariables = document.plan.base_variables as Record<string, number>;
    const snapshot = document.metadata.resolved_variables.find(
      (item) => item.variable_key === LAUNCH_AZIMUTH,
    );

    expect(Object.is(baseVariables[LAUNCH_AZIMUTH], -0)).toBe(false);
    expect(Object.is(snapshot?.value, -0)).toBe(false);
    expect(makeVariationExecutionMetadata(negative).planSha256).toBe(
      makeVariationExecutionMetadata(positive).planSha256,
    );
  });

  it("keeps maximum-safe seeds lossless and rejects unsafe plan integers", () => {
    const maximum = plan(Number.MAX_SAFE_INTEGER);
    const preceding = plan(Number.MAX_SAFE_INTEGER - 1);

    expect(makeVariationExecutionMetadata(maximum).planSha256).not.toBe(
      makeVariationExecutionMetadata(preceding).planSha256,
    );
    expect(() => makeVariationExecutionMetadata(plan(Number.MAX_SAFE_INTEGER + 1))).toThrow(
      /safe integer/i,
    );
    expect(() => makeVariationExecutionMetadata({
      ...plan(), nRuns: Number.MAX_SAFE_INTEGER + 1,
    })).toThrow(/safe integer/i);
    const firstUnsafe = 9_007_199_254_740_992;
    const collidingUnsafe = Number("9007199254740993");
    expect(collidingUnsafe).toBe(firstUnsafe);
    expect(() => makeVariationExecutionMetadata(plan(firstUnsafe))).toThrow(/safe integer/i);
    expect(() => makeVariationExecutionMetadata(plan(collidingUnsafe))).toThrow(/safe integer/i);
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
    ["RNG identity", (value) => { value.metadata.rng_identity.algorithm_id = "other"; }],
    ["implementation identity", (value) => { value.metadata.implementation_identity.solver_id = "other"; }],
    ["provenance digest", (value) => { value.provenance.producer_id = "rate-of-closure/other"; }],
    ["rng_identity fields", (value) => {
      Object.assign(value.metadata.rng_identity, { unexpected: true });
    }],
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
