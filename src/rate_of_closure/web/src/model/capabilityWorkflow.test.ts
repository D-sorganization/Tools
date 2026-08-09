import { describe, expect, it } from "vitest";

import {
  CAPABILITY_WORKFLOW_SCHEMA_VERSION,
  buildCapabilityWorkflow,
  capabilityWorkflowFromJson,
  capabilityWorkflowToJson,
  defaultCapabilityWorkflowInputs,
} from "./capabilityWorkflow";

describe("capability workflow", () => {
  it("builds a model-ready and auditable default driver workflow", () => {
    const document = buildCapabilityWorkflow(defaultCapabilityWorkflowInputs());

    expect(document.schemaVersion).toBe(CAPABILITY_WORKFLOW_SCHEMA_VERSION);
    expect(document.request.clubIds).toEqual(["driver"]);
    expect(document.profile.clubs[0].parameters[0].parameterId).toBe("ball_speed");
    expect(document.request.target.distanceM).toBe(230);
    expect(document.evaluatorConfig.spinDefaults[0].provenance).toContain("user-authored");
  });

  it("round-trips strict nested profile, request, and evaluator contracts", () => {
    const source = buildCapabilityWorkflow({
      ...defaultCapabilityWorkflowInputs(),
      clubId: "driver-fit-a",
      targetDistanceM: 245,
      targetLateralM: -4,
      totalSpinRpm: 2250,
      spinAxisTiltDeg: -3.5,
      candidateBudget: 4,
      ensembleSize: 6,
    });

    const encoded = capabilityWorkflowToJson(source);
    expect(capabilityWorkflowFromJson(encoded)).toEqual(source);
    expect(JSON.parse(encoded).schema_version).toBe(CAPABILITY_WORKFLOW_SCHEMA_VERSION);
  });

  it.each([
    [{ ballSpeedMps: 0 }, "ballSpeedMps"],
    [{ totalSpinRpm: -1 }, "totalSpinRpm"],
    [{ totalSpinRpm: 20_001 }, "totalSpinRpm"],
    [{ maxTimeS: 0 }, "maxTimeS"],
    [{ maxTimeS: 121 }, "maxTimeS"],
    [{ trajectorySampleIntervalS: 0.0015 }, "align"],
    [{ seed: 2 ** 31 }, "seed"],
    [{ candidateBudget: 501, ensembleSize: 201 }, "100000"],
    [{ alternativesCount: 3, candidateBudget: 2 }, "alternativesCount"],
    [{ spinAxisTiltDeg: 91 }, "spinAxisTiltDeg"],
  ] as const)("rejects unsafe or unrenderable input %o", (change, message) => {
    expect(() => buildCapabilityWorkflow({
      ...defaultCapabilityWorkflowInputs(), ...change,
    })).toThrow(message);
  });

  it("rejects extra document fields", () => {
    const parsed = JSON.parse(capabilityWorkflowToJson(
      buildCapabilityWorkflow(defaultCapabilityWorkflowInputs()),
    ));
    expect(() => capabilityWorkflowFromJson(JSON.stringify({
      ...parsed, unexpected: true,
    }))).toThrow("fields");
  });

  it("rejects a spin default bound to a different club", () => {
    const parsed = JSON.parse(capabilityWorkflowToJson(
      buildCapabilityWorkflow(defaultCapabilityWorkflowInputs()),
    ));
    parsed.evaluator_config.spin_defaults[0].club_id = "other-club";

    expect(() => capabilityWorkflowFromJson(JSON.stringify(parsed)))
      .toThrow("spin default clubIds");
  });
});
