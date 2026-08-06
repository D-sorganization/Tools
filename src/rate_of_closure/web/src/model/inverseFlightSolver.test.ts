import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/inverse_flight_solver_golden_v1.json";
import {
  parseInverseFlightRequest,
  parseInverseFlightResult,
  stableInverseFlightRequestJson,
  stableInverseFlightResultJson,
  type InverseFlightRequest,
  type SolverEvaluation,
} from "./inverseFlightContract";
import { solveInverseFlight } from "./inverseFlightSolver";

const request = (): InverseFlightRequest => parseInverseFlightRequest(fixture.request);

const analyticEvaluator = (parameters: Readonly<Record<string, number>>): SolverEvaluation => ({
  status: "complete",
  metrics: [
    { metricId: "carry_distance", value: 10 * parameters.speed, provenance: "analytic.carry" },
    { metricId: "apex_height", value: parameters.speed, provenance: "analytic.apex" },
  ],
  reason: null,
});

describe("desired-flight inverse solver", () => {
  it("returns deterministic feasible-first ranked candidates", () => {
    const result = solveInverseFlight(request(), analyticEvaluator);

    expect(result.status).toBe("solved");
    expect(result.evaluationsCompleted).toBe(5);
    expect(result.candidates).toHaveLength(3);
    expect(result.candidates.map((candidate) => candidate.rank)).toEqual([1, 2, 3]);
    expect(result.candidates[0].parameters[0].value).toBeCloseTo(7.5, 12);
    expect(result.candidates[0].feasible).toBe(true);
    expect(stableInverseFlightResultJson(result))
      .toBe(stableInverseFlightResultJson(solveInverseFlight(request(), analyticEvaluator)));
  });

  it("rejects unit drift, ineligible metrics, and duplicate objectives", () => {
    expect(() => parseInverseFlightRequest({
      ...fixture.request,
      objectives: [{ ...fixture.request.objectives[0], unit: "yd" }],
    })).toThrow(/canonical unit/);
    expect(() => parseInverseFlightRequest({
      ...fixture.request,
      objectives: [{ ...fixture.request.objectives[0], metric_id: "initial_velocity", unit: "m/s" }],
    })).toThrow(/not solver-eligible/);
    expect(() => parseInverseFlightRequest({
      ...fixture.request,
      objectives: [fixture.request.objectives[0], fixture.request.objectives[0]],
    })).toThrow(/unique/);
    expect(() => parseInverseFlightRequest({
      ...fixture.request, max_evaluations: 2.5, candidate_count: 1,
    })).toThrow(/integer/);
  });

  it("reports static infeasibility, no-impact, and nonconvergence", () => {
    const impossible = parseInverseFlightRequest({
      ...fixture.request,
      objectives: [{
        ...fixture.request.objectives[0], target_value: 100,
        lower_bound: 0, upper_bound: 50,
      }],
    });
    expect(solveInverseFlight(impossible, analyticEvaluator)).toMatchObject({
      status: "infeasible", evaluationsAttempted: 0,
      terminationReason: "target_outside_objective_bounds",
    });
    const noImpact = solveInverseFlight(request(), () => ({
      status: "no_impact", metrics: [], reason: "club_did_not_contact_ball",
    }));
    expect(noImpact).toMatchObject({ status: "no_impact", noImpactCount: 5 });
    const nonconverged = solveInverseFlight(request(), () => ({
      status: "nonconverged", metrics: [], reason: "integrator_budget_exhausted",
    }));
    expect(nonconverged).toMatchObject({ status: "nonconverged", failedCount: 5 });
    const malformed = solveInverseFlight(request(), () => ({
      status: "complete", metrics: [
        { metricId: "carry_distance", value: Number.NaN, provenance: "invalid" },
      ], reason: null,
    }));
    expect(malformed).toMatchObject({ status: "nonconverged", failedCount: 5 });
  });

  it("matches Python's strict deterministic result fixture", async () => {
    expect(JSON.parse(stableInverseFlightRequestJson(request()))).toEqual(fixture.request);
    const result = solveInverseFlight(request(), analyticEvaluator);
    const serialized = stableInverseFlightResultJson(result);
    expect(stableInverseFlightResultJson(parseInverseFlightResult(JSON.parse(serialized))))
      .toBe(serialized);
    const invalid = JSON.parse(serialized);
    invalid.candidates[0].unexpected = true;
    expect(() => parseInverseFlightResult(invalid)).toThrow(/solution candidate fields/);
    const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(serialized));
    const hex = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
    expect(hex).toBe(fixture.result_sha256);
  });
});
