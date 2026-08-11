import { describe, expect, it } from "vitest";

import parserCasesFixture from "./__fixtures__/capability_workflow_parser_cases_v1.json";

import {
  CAPABILITY_WORKFLOW_SCHEMA_VERSION,
  buildCapabilityWorkflow,
  capabilityWorkflowFromJson,
  capabilityWorkflowToJson,
  defaultCapabilityWorkflowInputs,
} from "./capabilityWorkflow";

interface ParserCase {
  readonly id: string;
  readonly path: readonly (string | number)[];
  readonly value: unknown;
  readonly accepted: boolean;
}

const parserCases = parserCasesFixture.cases as readonly ParserCase[];
const hostileNumbers = parserCasesFixture.hostile_numbers;

const mutatedWorkflow = (testCase: ParserCase): string => {
  const payload: unknown = JSON.parse(capabilityWorkflowToJson(
    buildCapabilityWorkflow(defaultCapabilityWorkflowInputs()),
  ));
  let cursor = payload;
  testCase.path.slice(0, -1).forEach((key) => {
    cursor = (cursor as Record<string | number, unknown>)[key];
  });
  const terminal = testCase.path[testCase.path.length - 1];
  if (terminal === undefined) throw new RangeError("fixture path must be nonempty");
  (cursor as Record<string | number, unknown>)[terminal] = testCase.value;
  return JSON.stringify(payload);
};

describe("capability workflow", () => {
  it("uses the supported shared parser fixture schema", () => {
    expect(parserCasesFixture.schema_version)
      .toBe("capability-workflow-parser-cases/v1");
  });

  it.each(hostileNumbers)("rejects shared hostile number $id", (testCase) => {
    const source = capabilityWorkflowToJson(
      buildCapabilityWorkflow(defaultCapabilityWorkflowInputs()),
    );
    const rawNumber = testCase.digit.repeat(testCase.digits);
    const hostile = source.replace(
      '"candidate_budget":8',
      `"candidate_budget":${rawNumber}`,
    );

    expect(() => capabilityWorkflowFromJson(hostile)).toThrow(/magnitude|finite/i);
  });

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

  it.each(parserCases)("applies shared strict parser case $id", (testCase) => {
    const parse = () => capabilityWorkflowFromJson(mutatedWorkflow(testCase));

    if (testCase.accepted) expect(parse).not.toThrow();
    else expect(parse).toThrow();
  });
});
