import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/capability_optimizer_golden_v1.json";
import {
  parseOptimizationRequest,
  parseOptimizationResult,
  parsePlayerCapabilityProfile,
  type SolverEvaluation,
} from "./capabilityContract";
import { optimizeCapability } from "./capabilityOptimizer";

const evaluator = (clubId: string, parameters: Readonly<Record<string, number>>): SolverEvaluation => {
  const clubBonus = clubId === "driver" ? 18 : 0;
  return {
    status: "complete",
    metrics: [
      { metricId: "carry_distance", value: parameters.ball_speed * 2 + parameters.launch_angle * 0.4 + clubBonus, provenance: "analytic.carry" },
      { metricId: "carry_offline", value: parameters.launch_direction * 1.5, provenance: "analytic.offline" },
    ],
    reason: null,
  };
};

describe("capability optimizer", () => {
  it("matches the Python parity fixture", () => {
    const profile = parsePlayerCapabilityProfile(fixture.profile);
    const request = parseOptimizationRequest(fixture.request);
    const result = optimizeCapability(profile, request, evaluator);

    expect(result.alternatives[0].clubId).toBe(fixture.expected.best_club_id);
    expect(result.alternatives[0].meanCarryM).toBeCloseTo(fixture.expected.mean_carry_m, 10);
    expect(result.alternatives[0].targetHoldProbability).toBeCloseTo(fixture.expected.target_hold_probability, 10);
    expect(parseOptimizationResult(JSON.parse(JSON.stringify(result)))).toEqual(result);
  });

  it("supports every robust objective deterministically", () => {
    const profile = parsePlayerCapabilityProfile(fixture.profile);
    const base = fixture.request;
    (["maximize_carry", "minimize_expected_miss", "maximize_target_hold", "distance_control_pareto"] as const)
      .forEach((objective) => {
        const request = parseOptimizationRequest({ ...base, objective });
        const first = optimizeCapability(profile, request, evaluator);
        const second = optimizeCapability(profile, request, evaluator);
        expect(first).toEqual(second);
        expect(first.alternatives.length).toBeGreaterThan(0);
        expect(first.alternatives[0].cvarMissM).toBeGreaterThanOrEqual(first.alternatives[0].expectedMissM);
        if (objective === "distance_control_pareto") expect(first.alternatives[0].paretoEfficient).toBe(true);
      });
  });

  it("fails closed for invalid covariance, unknown fields, and no-impact ensembles", () => {
    const invalidProfile = structuredClone(fixture.profile);
    invalidProfile.clubs[0].matrix = [[1, 2, 0], [2, 1, 0], [0, 0, 1]];
    expect(() => parsePlayerCapabilityProfile(invalidProfile)).toThrow(/positive semidefinite/);

    const profile = parsePlayerCapabilityProfile(fixture.profile);
    const request = parseOptimizationRequest(fixture.request);
    const result = optimizeCapability(profile, request, () => ({
      status: "no_impact", metrics: [], reason: "missed_ball",
    }));
    expect(result).toMatchObject({ status: "nonconverged", noImpactCount: 120 });
    expect(() => parseOptimizationResult({ ...result, unexpected: true }))
      .toThrow(/fields do not match/);
  });
});
