import { expect, it, vi } from "vitest";

import fixture from "./__fixtures__/capability_optimizer_golden_v1.json";
import {
  parseOptimizationRequest,
  parsePlayerCapabilityProfile,
  type CapabilityEvaluator,
  type SolverEvaluation,
} from "./capabilityContract";
import {
  CapabilityOptimizationCancelled,
  capabilitySampleObservationWire,
  optimizeCapability,
  type CapabilitySampleObservation,
} from "./capabilityOptimizer";

const complete = (carry = 200, offline = 2): SolverEvaluation => ({
  status: "complete",
  metrics: [
    { metricId: "carry_distance", value: carry, provenance: "test.carry" },
    { metricId: "carry_offline", value: offline, provenance: "test.offline" },
    { metricId: "apex_height", value: 31, provenance: "test.apex" },
  ],
  reason: null,
});

const setup = (candidateBudget: number, ensembleSize: number, clubIds?: readonly string[]) => ({
  profile: parsePlayerCapabilityProfile(fixture.profile),
  request: parseOptimizationRequest({
    ...fixture.request,
    candidate_budget: candidateBudget,
    ensemble_size: ensembleSize,
    alternatives_count: 1,
    club_ids: clubIds ?? fixture.request.club_ids,
  }),
});

it("emits immutable samples in exact round-robin and ensemble order", () => {
    const { profile, request } = setup(3, 2);
    const observations: CapabilitySampleObservation[] = [];

    optimizeCapability(profile, request, () => complete(), {
      observationSink: (observation) => observations.push(observation),
    });

    expect(observations.map((item) => [
      item.attemptOrdinal,
      item.attemptedCount,
      item.totalCount,
      item.candidateOrdinal,
      item.clubCandidateOrdinal,
      item.sampleOrdinal,
      item.clubId,
    ])).toEqual([
      [0, 1, 6, 0, 0, 0, "iron-7"],
      [1, 2, 6, 0, 0, 1, "iron-7"],
      [2, 3, 6, 1, 0, 0, "driver"],
      [3, 4, 6, 1, 0, 1, "driver"],
      [4, 5, 6, 2, 1, 0, "iron-7"],
      [5, 6, 6, 2, 1, 1, "iron-7"],
    ]);
    expect(observations[0].parameters.map((item) => item.parameterId))
      .toEqual(profile.clubs[0].parameters.map((item) => item.parameterId));
    expect(observations[0].metrics.map((item) => item.metricId))
      .toEqual(["carry_distance", "carry_offline", "apex_height"]);
    expect(observations[0]).toMatchObject({
      schemaVersion: "capability-sample-observation/v1",
      problemId: request.problemId,
      sourceStatus: "complete",
      effectiveStatus: "complete",
      reasonCode: null,
      sourceReason: null,
    });
  });

it("freezes observations and serializes the exact cross-runtime wire shape", () => {
    const { profile, request } = setup(1, 1);
    const observations: CapabilitySampleObservation[] = [];
    optimizeCapability(profile, request, () => complete(), {
      observationSink: (observation) => observations.push(observation),
    });

    expect(Object.isFrozen(observations[0])).toBe(true);
    expect(Object.isFrozen(observations[0].parameters)).toBe(true);
    expect(Object.isFrozen(observations[0].parameters[0])).toBe(true);
    expect(Object.isFrozen(observations[0].metrics)).toBe(true);
    expect(Object.isFrozen(observations[0].metrics[0])).toBe(true);
    const wire = capabilitySampleObservationWire(observations[0]);
    expect(Object.keys(wire)).toEqual([
      "schema_version", "problem_id", "attempt_ordinal", "attempted_count", "total_count",
      "candidate_ordinal", "club_candidate_ordinal", "sample_ordinal", "club_id", "parameters",
      "source_status", "effective_status", "reason_code", "source_reason", "metrics",
    ]);
    expect(wire).toMatchObject({
      schema_version: "capability-sample-observation/v1",
      problem_id: request.problemId,
      attempt_ordinal: 0,
      attempted_count: 1,
      total_count: 1,
      candidate_ordinal: 0,
      club_candidate_ordinal: 0,
      sample_ordinal: 0,
      club_id: "iron-7",
      source_status: "complete",
      effective_status: "complete",
      reason_code: null,
      source_reason: null,
    });
    expect(wire.parameters[0])
      .toEqual({
        parameter_id: observations[0].parameters[0].parameterId,
        unit: observations[0].parameters[0].unit,
        nominal_value: observations[0].parameters[0].nominalValue,
        perturbed_value: observations[0].parameters[0].perturbedValue,
      });
    expect(wire.metrics[0]).toEqual({
      metric_id: "carry_distance", value: 200, provenance: "test.carry",
    });
  });

it("normalizes all evaluator statuses, exceptions, and malformed results", () => {
    const { profile, request } = setup(7, 1, ["iron-7"]);
    const outcomes: unknown[] = [
      complete(),
      { status: "no_impact", metrics: [], reason: "missed_ball" },
      { status: "failed", metrics: [], reason: "solver_failed" },
      { status: "nonconverged", metrics: [], reason: "iteration_limit" },
      new Error("evaluator exploded"),
      { unexpected: true },
      { status: "complete", metrics: [{ metricId: "carry_distance", value: 200, provenance: "test" }], reason: null },
    ];
    const observations: CapabilitySampleObservation[] = [];
    let index = 0;
    const evaluator = (() => {
      const outcome = outcomes[index++];
      if (outcome instanceof Error) throw outcome;
      return outcome;
    }) as CapabilityEvaluator;

    const result = optimizeCapability(profile, request, evaluator, {
      observationSink: (observation) => observations.push(observation),
    });

    expect(observations.map((item) => [
      item.sourceStatus, item.effectiveStatus, item.reasonCode, item.sourceReason,
    ])).toEqual([
      ["complete", "complete", null, null],
      ["no_impact", "no_impact", "missed_ball", "missed_ball"],
      ["failed", "failed", "solver_failed", "solver_failed"],
      ["nonconverged", "failed", "iteration_limit", "iteration_limit"],
      [null, "failed", "evaluator_exception", null],
      [null, "failed", "invalid_evaluator_result", null],
      ["complete", "failed", "missing_required_landing_metrics", null],
    ]);
    expect(result).toMatchObject({ evaluationsCompleted: 1, noImpactCount: 1, failedCount: 5 });
  });

it("rejects every semantic SolverEvaluation contract violation", () => {
    const { profile, request } = setup(5, 1, ["iron-7"]);
    const metric = { metricId: "carry_distance", value: 200, provenance: "test" };
    const outcomes: unknown[] = [
      { status: "complete", metrics: [metric, metric], reason: null },
      { status: "complete", metrics: [metric], reason: "unexpected" },
      { status: "no_impact", metrics: [metric], reason: "missed_ball" },
      { status: "failed", metrics: [], reason: null },
      { status: "nonconverged", metrics: [], reason: "  " },
    ];
    const observations: CapabilitySampleObservation[] = [];
    let index = 0;
    const evaluator = (() => outcomes[index++]) as CapabilityEvaluator;

    const result = optimizeCapability(profile, request, evaluator, {
      observationSink: (observation) => observations.push(observation),
    });

    expect(observations).toHaveLength(5);
    expect(observations.every((observation) =>
      observation.sourceStatus === null
      && observation.effectiveStatus === "failed"
      && observation.reasonCode === "invalid_evaluator_result"
      && observation.sourceReason === null
      && observation.metrics.length === 0)).toBe(true);
    expect(result).toMatchObject({
      status: "nonconverged", evaluationsCompleted: 0, noImpactCount: 0, failedCount: 5,
    });
  });

it("cancels before the next evaluator call with typed progress", () => {
    const { profile, request } = setup(4, 2);
    const evaluator = vi.fn(() => complete());
    const observations: CapabilitySampleObservation[] = [];

    expect(() => optimizeCapability(profile, request, evaluator, {
      observationSink: (observation) => observations.push(observation),
      shouldCancel: () => evaluator.mock.calls.length >= 3,
    })).toThrow(CapabilityOptimizationCancelled);
    expect(evaluator).toHaveBeenCalledTimes(3);
    expect(observations).toHaveLength(3);
    try {
      optimizeCapability(profile, request, evaluator, { shouldCancel: () => true });
    } catch (error) {
      expect(error).toMatchObject({ attemptedCount: 0, totalCount: 8 });
    }
    expect(() => new CapabilityOptimizationCancelled(-1, 8)).toThrow(/nonnegative integer/);
    expect(() => new CapabilityOptimizationCancelled(3, 2)).toThrow(/not less than/);
  });

it("propagates sink errors and leaves the compact result invariant", () => {
    const { profile, request } = setup(2, 2);
    const baseline = optimizeCapability(profile, request, () => complete());
    const observations: CapabilitySampleObservation[] = [];
    const observed = optimizeCapability(profile, request, () => complete(), {
      observationSink: (observation) => observations.push(observation),
    });

    expect(observed).toEqual(baseline);
    expect("observations" in observed).toBe(false);
    expect(observations).toHaveLength(4);

    const sentinel = new Error("sink failed");
    const evaluator = vi.fn(() => complete());
    expect(() => optimizeCapability(profile, request, evaluator, {
      observationSink: () => { throw sentinel; },
    })).toThrow(sentinel);
    expect(evaluator).toHaveBeenCalledTimes(1);
  });
