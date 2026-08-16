import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/impact_solution_families_golden_v1.json";
import {
  parseImpactSolutionRequest,
  parseImpactSolutionResult,
  stableImpactSolutionRequestJson,
  stableImpactSolutionResultJson,
} from "./impactSolutionContract";

describe("impact solution family contracts", () => {
  it("parses and serializes the shared Python fixture", () => {
    const request = parseImpactSolutionRequest(fixture.request);
    const result = parseImpactSolutionResult(fixture.result);

    expect(JSON.parse(stableImpactSolutionRequestJson(request))).toEqual(fixture.request);
    expect(JSON.parse(stableImpactSolutionResultJson(result))).toEqual(fixture.result);
  });

  it("fails closed on frame, unit, and unsupported-variable drift", () => {
    expect(() => parseImpactSolutionRequest({
      ...fixture.request,
      target_frame_id: "unknown",
    })).toThrow(/target_frame_id/);
    expect(() => parseImpactSolutionRequest({
      ...fixture.request,
      inverse_request: {
        ...fixture.request.inverse_request,
        variables: [{ ...fixture.request.inverse_request.variables[0], unit: "mph" }],
      },
    })).toThrow(/canonical unit/);
    expect(() => parseImpactSolutionRequest({
      ...fixture.request,
      inverse_request: {
        ...fixture.request.inverse_request,
        variables: [{
          ...fixture.request.inverse_request.variables[0],
          parameter_id: "shaft_flex",
        }],
      },
    })).toThrow(/unsupported delivery variable/);
    expect(() => parseImpactSolutionRequest({
      ...fixture.request,
      inverse_request: {
        ...fixture.request.inverse_request,
        variables: [{ ...fixture.request.inverse_request.variables[0], upper_bound: 120 }],
      },
    })).toThrow(/supported range/);
  });

  it("rejects undeclared result fields and invalid family ranks", () => {
    expect(() => parseImpactSolutionResult({
      ...fixture.result,
      unexpected: true,
    })).toThrow(/fields/);
    expect(() => parseImpactSolutionResult({
      ...fixture.result,
      families: [{ ...fixture.result.families[0], rank: 2 }],
    })).toThrow(/contiguous/);
    expect(() => parseImpactSolutionResult({
      ...fixture.result,
      evaluations_attempted: 2,
      rejected_candidates: [{
        evaluation_index: 0,
        parameters: fixture.result.families[0].members[0].parameters,
        reason: "duplicate",
        status: "complete",
      }],
    })).toThrow(/exactly once/);
    expect(() => parseImpactSolutionResult({
      ...fixture.result,
      evaluations_attempted: true,
    })).toThrow(/finite/);
    expect(() => parseImpactSolutionResult({
      ...fixture.result,
      families: [{
        ...fixture.result.families[0],
        representative_evaluation_index: -1,
      }],
    })).toThrow(/representative_evaluation_index/);
  });
});
