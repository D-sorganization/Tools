/** Contract tests for the model-backed capability evaluator. */

import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/capability_optimizer_golden_v1.json";
import parityFixture from "./__fixtures__/capability_flight_evaluator_parity_v1.json";
import {
  parseOptimizationRequest,
  parsePlayerCapabilityProfile,
} from "./capabilityContract";
import { makeCapabilityFlightEvaluator } from "./capabilityFlightEvaluator";
import { optimizeCapability } from "./capabilityOptimizer";

const PINNED = Object.freeze({
  ball_speed: 74.65568,
  launch_angle: 10.9,
  launch_direction: 0.0,
});
const SPIN_DEFAULTS = Object.freeze([Object.freeze({
  clubId: "driver", totalSpinRpm: 2686, spinAxisTiltDeg: 0,
  provenance: "test-fixture",
})]);

const parameter = (parameterId: string, unit: string, baseline: number,
  lowerBound: number, upperBound: number) => ({
  parameter_id: parameterId, unit, baseline, bias: 0, standard_deviation: 0,
  lower_bound: lowerBound, upper_bound: upperBound,
  evidence_lower_bound: lowerBound, evidence_upper_bound: upperBound,
});

const profile = (includeSpin = false) => {
  const parameters = [
    parameter("ball_speed", "m/s", PINNED.ball_speed, 40, 80),
    parameter("launch_angle", "deg", PINNED.launch_angle, 0, 30),
    parameter("launch_direction", "deg", 0, -15, 15),
  ];
  if (includeSpin) parameters.push(
    parameter("total_spin", "rpm", 2686, 0, 6000),
    parameter("spin_axis_tilt", "deg", 0, -45, 45),
  );
  const matrix = parameters.map((_, row) =>
    parameters.map((__, column) => row === column ? 1 : 0));
  return parsePlayerCapabilityProfile({
    schema_version: "player-capability-profile/v1", profile_id: "player",
    clubs: [{ club_id: "driver", parameters, matrix_kind: "correlation",
      matrix, provenance: "test", confidence: 1 }],
    provenance: "test", confidence: 1,
  });
};

const request = () => parseOptimizationRequest({
  schema_version: "capability-optimization-request/v1", problem_id: "flight-evaluator",
  objective: "maximize_target_hold", club_ids: ["driver"],
  target: { kind: "green", distance_m: 247, lateral_m: 0, radius_m: 10,
    band_half_length_m: 15, half_width_m: 15 },
  candidate_budget: 1, ensemble_size: 1, alternatives_count: 1,
  seed: 7, cvar_alpha: 0.8, minimum_success_fraction: 0.5,
});

const metricMap = (evaluation: ReturnType<ReturnType<typeof makeCapabilityFlightEvaluator>>) =>
  new Map(evaluation.metrics.map((item) => [item.metricId, item]));

describe("capability flight evaluator", () => {
  it("runs Waterloo/Penner with every available scalar and target value", () => {
    const evaluation = makeCapabilityFlightEvaluator(
      profile(), request(), { spinDefaults: SPIN_DEFAULTS },
    )("driver", PINNED);
    const metrics = metricMap(evaluation);

    expect(evaluation.status).toBe("complete");
    expect(metrics.get("ball_speed")?.value).toBeCloseTo(74.65568, 5);
    expect(metrics.get("vertical_launch_angle")?.value).toBeCloseTo(10.9, 8);
    expect(metrics.get("carry_distance")?.value).toBeGreaterThan(247.484 * 0.99);
    expect(metrics.get("carry_distance")?.value).toBeLessThan(247.484 * 1.01);
    expect(metrics.has("target_residual")).toBe(true);
    expect(metrics.has("initial_velocity")).toBe(false);
    expect(metrics.size).toBe(16);
    expect(evaluation.metrics.every((item) => item.provenance
      .includes("waterloo_penner:waterloo-penner-coefficients/v1"))).toBe(true);
    expect(evaluation.metrics.every((item) => item.provenance
      .includes("spin:fixed_club_default:test-fixture"))).toBe(true);
    for (const [metricId, expected] of Object.entries(
      parityFixture.expected_scalars,
    )) {
      const value = evaluation.metrics.find(
        (item) => item.metricId === metricId,
      )?.value as number;
      expect(Math.abs(value - expected.value))
        .toBeLessThanOrEqual(expected.absolute_tolerance);
    }
  });

  it("preserves positive-right direction and fade-side spin tilt", () => {
    const evaluator = makeCapabilityFlightEvaluator(profile(true), request());
    const direction = evaluator("driver", {
      ...PINNED, launch_direction: 5, total_spin: 2686, spin_axis_tilt: 0,
    });
    const tilt = evaluator("driver", {
      ...PINNED, total_spin: 2686, spin_axis_tilt: 10,
    });
    const negativeTilt = evaluator("driver", {
      ...PINNED, total_spin: 2686, spin_axis_tilt: -10,
    });
    const directionMetrics = metricMap(direction);
    const tiltMetrics = metricMap(tilt);

    expect(directionMetrics.get("launch_direction")?.value).toBeCloseTo(5, 8);
    expect(directionMetrics.get("carry_offline")?.value).toBeGreaterThan(1);
    expect(tiltMetrics.get("spin_axis_tilt")?.value).toBeCloseTo(10, 8);
    expect(tiltMetrics.get("carry_offline")?.value).toBeGreaterThan(1);
    expect(metricMap(negativeTilt).get("carry_offline")?.value).toBeLessThan(-1);
  });

  it("plugs into the existing profile fixture and optimizer", () => {
    const parsedProfile = parsePlayerCapabilityProfile(fixture.profile);
    const parsedRequest = parseOptimizationRequest({
      ...fixture.request, club_ids: ["driver"], candidate_budget: 1,
      ensemble_size: 1, alternatives_count: 1,
    });
    const result = optimizeCapability(
      parsedProfile,
      parsedRequest,
      makeCapabilityFlightEvaluator(
        parsedProfile, parsedRequest, { spinDefaults: SPIN_DEFAULTS },
      ),
    );

    expect(result.status).toBe("solved");
    expect(result.evaluationsCompleted).toBe(1);
    expect(result.alternatives[0].meanCarryM).toBeGreaterThan(100);
  });

  it("types short-horizon nonconvergence and supports zero spin", () => {
    const short = makeCapabilityFlightEvaluator(
      profile(), request(), { maxTimeS: 0.001, spinDefaults: SPIN_DEFAULTS },
    );
    expect(short("driver", PINNED)).toEqual({
      status: "nonconverged", metrics: [],
      reason: "no_ground_crossing_before_max_time",
    });
    const zeroSpin = makeCapabilityFlightEvaluator(profile(), request(), {
      spinDefaults: [{ ...SPIN_DEFAULTS[0], totalSpinRpm: 0 }],
    })("driver", PINNED);
    expect(zeroSpin.status).toBe("complete");
    expect(metricMap(zeroSpin).has("spin_axis_tilt")).toBe(false);
    const coarse = makeCapabilityFlightEvaluator(profile(), request(), {
      trajectorySampleIntervalS: 0.1, spinDefaults: SPIN_DEFAULTS,
    })("driver", PINNED);
    expect(coarse.status).toBe("complete");
  });

  it("requires an auditable spin default for each club", () => {
    const driverProfile = profile();
    const iron = { ...driverProfile.clubs[0], clubId: "iron" };
    const twoClubProfile = {
      ...driverProfile, clubs: [...driverProfile.clubs, iron],
    };
    const twoClubRequest = {
      ...request(), clubIds: ["driver", "iron"],
    };
    expect(() => makeCapabilityFlightEvaluator(
      twoClubProfile, twoClubRequest, { spinDefaults: SPIN_DEFAULTS },
    )).toThrow(/iron.*explicit spin default/);

    const evaluator = makeCapabilityFlightEvaluator(
      twoClubProfile, twoClubRequest, { spinDefaults: [
        { ...SPIN_DEFAULTS[0], provenance: "driver-fixture" },
        { clubId: "iron", totalSpinRpm: 6200, spinAxisTiltDeg: 0,
          provenance: "iron-fixture" },
      ] },
    );
    const result = evaluator("iron", PINNED);
    expect(result.status).toBe("complete");
    expect(result.metrics.every((item) =>
      item.provenance.includes("spin:fixed_club_default:iron-fixture"))).toBe(true);
  });

  it("rejects schema, units, bounds, and invalid configuration", () => {
    const evaluator = makeCapabilityFlightEvaluator(
      profile(), request(), { spinDefaults: SPIN_DEFAULTS },
    );
    expect(() => evaluator("driver", { ...PINNED, unused: 1 })).toThrow("fields");
    expect(() => evaluator("driver", { ...PINNED, ball_speed: 100 })).toThrow("safe bounds");
    const invalid = structuredClone(profile());
    const club = invalid.clubs[0] as unknown as { parameters: { unit: string }[] };
    club.parameters[0].unit = "mph";
    expect(() => makeCapabilityFlightEvaluator(
      invalid, request(), { spinDefaults: SPIN_DEFAULTS },
    )).toThrow(/ball_speed.*m\/s/);
    expect(() => makeCapabilityFlightEvaluator(profile(), request(), { maxTimeS: 0 }))
      .toThrow("maxTimeS");
    expect(() => makeCapabilityFlightEvaluator(profile(), request()))
      .toThrow("explicit spin default");
    expect(() => makeCapabilityFlightEvaluator(profile(), request(), {
      trajectorySampleIntervalS: 9, spinDefaults: SPIN_DEFAULTS,
    })).toThrow(/\[0.001, 0.1\]/);
    expect(() => makeCapabilityFlightEvaluator(profile(), request(), {
      trajectorySampleIntervalS: 0.0015, spinDefaults: SPIN_DEFAULTS,
    })).toThrow("align");
  });

  it.each([
    ["ball_speed", "lowerBound", 0, /ball_speed.*greater than zero/],
    ["launch_angle", "upperBound", 100, /launch_angle.*\[-90, 90\]/],
    ["launch_direction", "upperBound", 300, /launch_direction.*\[-180, 180\]/],
    ["total_spin", "lowerBound", -1, /total_spin.*\[0, Infinity\]/],
    ["spin_axis_tilt", "upperBound", 120, /spin_axis_tilt.*\[-90, 90\]/],
  ])("rejects %s profiles outside the physical domain", (
    parameterId, field, value, message,
  ) => {
    const invalid = structuredClone(profile(
      parameterId === "total_spin" || parameterId === "spin_axis_tilt",
    ));
    const parameters = invalid.clubs[0].parameters as unknown as
      Record<string, number | string>[];
    const selected = parameters.find((item) => item.parameterId === parameterId)!;
    selected[field as string] = value as number;
    expect(() => makeCapabilityFlightEvaluator(
      invalid, request(), { spinDefaults: SPIN_DEFAULTS },
    )).toThrow(message as RegExp);
  });
  // Bounds mirror the Shot Optimizer default profile, which permits the
  // descending launches the fixture profile's narrower envelope excludes.
  const wideProfile = () => {
    const parameters = [
      parameter("ball_speed", "m/s", 67, 1, 100),
      parameter("launch_angle", "deg", 12.5, -10, 45),
      parameter("launch_direction", "deg", 0, -30, 30),
    ];
    const matrix = parameters.map((_, row) =>
      parameters.map((__, column) => row === column ? 1 : 0));
    return parsePlayerCapabilityProfile({
      schema_version: "player-capability-profile/v1", profile_id: "player",
      clubs: [{ club_id: "driver", parameters, matrix_kind: "correlation",
        matrix, provenance: "test", confidence: 1 }],
      provenance: "test", confidence: 1,
    });
  };

  it.each([
    [-4.323746585003578, 83.2427975264564],
    [-3.169046046624327, 85.84087373780972],
  ])("types a descending launch at %s deg as nonconverged, not an exception", (
    launchAngle, ballSpeed,
  ) => {
    const evaluate = makeCapabilityFlightEvaluator(
      wideProfile(), request(), { spinDefaults: SPIN_DEFAULTS },
    );

    const evaluation = evaluate("driver", {
      ball_speed: ballSpeed, launch_angle: launchAngle, launch_direction: 9.2,
    });

    // The Python runtime reports these same samples as nonconverged. The
    // ground interpolation previously extrapolated to a negative time and
    // threw a RangeError that surfaced as an untyped evaluator_exception.
    expect(evaluation.status).toBe("nonconverged");
    expect(evaluation.metrics).toHaveLength(0);
  });
});
