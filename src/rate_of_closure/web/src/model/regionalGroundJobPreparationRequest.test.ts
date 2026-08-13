import { describe, expect, it } from "vitest";

import jobFixture from "./__fixtures__/regional_ground_execution_job_golden_v1.json";
import { canonicalGroundJson } from "./flightGroundContract";
import {
  MAX_REGIONAL_GROUND_JOB_PREPARATION_REQUEST_BYTES,
  parseRegionalGroundJobPreparationRequest,
  regionalGroundJobPreparationRequestFromJson,
  REGIONAL_GROUND_JOB_PREPARATION_REQUEST_SCHEMA,
  stableRegionalGroundJobPreparationRequestJson,
} from "./regionalGroundJobPreparationRequest";

const source = jobFixture.job;
const request = {
  schema_version: REGIONAL_GROUND_JOB_PREPARATION_REQUEST_SCHEMA,
  unit_system: "SI",
  job_id: "prepared-job-001",
  launch: source.launch,
  variation_request: source.variation_request,
};

describe("regional-ground job-preparation request wire", () => {
  it("parses, deeply freezes, and serializes the exact canonical editor snapshot", () => {
    const parsed = parseRegionalGroundJobPreparationRequest(request);
    const expected = canonicalGroundJson(request);

    expect(stableRegionalGroundJobPreparationRequestJson(parsed)).toBe(expected);
    expect(regionalGroundJobPreparationRequestFromJson(expected)).toEqual(parsed);
    expect(Object.isFrozen(parsed)).toBe(true);
    expect(Object.isFrozen(parsed.launch.ball_setup)).toBe(true);
    expect(Object.isFrozen(parsed.variation_request)).toBe(true);
    expect(Object.isFrozen(parsed.variation_request.variation_plan)).toBe(true);
  });

  it.each([
    ["extra root field", { ...request, unexpected: true }, /fields/i],
    ["wrong schema", { ...request, schema_version: "v2" }, /schema_version/i],
    ["wrong units", { ...request, unit_system: "US" }, /unit_system/i],
    ["unsafe job id", { ...request, job_id: "../escape" }, /job_id/i],
    ["Boolean launch number", {
      ...request,
      launch: { ...request.launch, ball_speed_m_s: true },
    }, /ball_speed/i],
    ["extra launch field", {
      ...request,
      launch: { ...request.launch, unexpected: true },
    }, /launch.*fields/i],
    ["invalid variation request", {
      ...request,
      variation_request: { ...request.variation_request, unexpected: true },
    }, /fields/i],
  ])("rejects %s", (_name, changed, message) => {
    expect(() => parseRegionalGroundJobPreparationRequest(changed)).toThrow(message);
  });

  it("rejects duplicate and oversized UTF-8 JSON before parsing", () => {
    const sourceText = stableRegionalGroundJobPreparationRequestJson(request);
    const duplicate = sourceText.replace(
      '"job_id":',
      '"job_id":"duplicate","job_id":',
    );

    expect(() => regionalGroundJobPreparationRequestFromJson(duplicate)).toThrow(/duplicate/i);
    expect(() => regionalGroundJobPreparationRequestFromJson(
      "\u00e9".repeat(MAX_REGIONAL_GROUND_JOB_PREPARATION_REQUEST_BYTES / 2 + 1),
    )).toThrow(/maximum wire size/i);
  });
});
