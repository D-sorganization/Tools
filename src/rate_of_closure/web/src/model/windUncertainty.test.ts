import fixture from "./__fixtures__/wind_uncertainty_golden_v1.json";
import riskFixture from "./__fixtures__/wind_strategy_risk_golden_v2.json";
import { describe, expect, it } from "vitest";

import { directLaunch } from "./flightExplorer";
import { meteorologicalWind } from "./wind";
import { summarizeStrategyOutcomes } from "./windStrategyMetrics";
import {
  analyzeWindStrategies,
  sampleWindTrials,
  type WindOutcomeStatus,
  type WindStrategyOutcome,
  type WindStrategyRequest,
  type WindUncertaintySpec,
} from "./windUncertainty";

const spec = (trials: number): WindUncertaintySpec => ({
  schema_version: "wind-uncertainty/v1",
  trials,
  seed: 4199,
  true_speed_mps: { kind: "normal", center: 5, spread: 1.1, minimum: 0 },
  true_from_bearing_deg: { kind: "uniform", center: 15, spread: 20 },
  estimate_error: {
    speed_bias_mps: -0.8,
    speed_std_mps: 0.7,
    bearing_bias_deg: 3,
    bearing_std_deg: 4,
    correlation: 0.45,
  },
  provenance: "test/weather_station_plus_player_estimate",
});

describe("sampleWindTrials", () => {
  it("matches the Python-readable golden fixture exactly", () => {
    expect(sampleWindTrials(spec(fixture.trials.length))).toEqual(fixture.trials);
  });

  it("is deterministic and rejects an invalid correlation", () => {
    expect(sampleWindTrials(spec(8))).toEqual(sampleWindTrials(spec(8)));
    expect(() => sampleWindTrials({
      ...spec(8),
      estimate_error: { ...spec(8).estimate_error, correlation: -1.1 },
    })).toThrow(/correlation/);
  });
});

describe("analyzeWindStrategies", () => {
  it("returns paired scatter outcomes and nonnegative CRN regret", () => {
    const launch = directLaunch({
      ballSpeedMph: 150,
      launchAngleDeg: 12,
      azimuthDeg: 0,
      spinRpm: 2500,
      spinAxisTiltDeg: 0,
    });
    const result = analyzeWindStrategies({
      uncertainty: spec(4),
      strategies: [
        { id: "straight", label: "Straight", launch, crosswind_aim_gain_rad_per_mps: 0 },
        {
          id: "compensated",
          label: "Compensated",
          launch,
          crosswind_aim_gain_rad_per_mps: 0.2 * Math.PI / 180,
        },
      ],
      target: { forward_m: 230, right_m: 0 },
      analysis: {
        model_name: "waterloo_penner",
        max_time_s: 10,
        time_step_s: 0.001,
        miss_scale_m: 20,
        failure_cost: 100,
        target_radius_m: 10,
        miss_distance_cvar_alpha: 0.75,
      },
    });

    expect(result.outcomes).toHaveLength(8);
    expect(result.summaries).toHaveLength(2);
    expect(result.schema_version).toBe("wind-strategy-analysis/v2");
    expect(result.summaries.every((item) => item.completed_trials === 4)).toBe(true);
    expect(result.summaries.every((item) => item.expected_regret >= 0)).toBe(true);
    result.summaries.forEach((item) => {
      expect(item.expected_regret).toBe(item.expected_preset_oracle_regret);
      expect(item.probability_best).toBe(item.preset_oracle_probability_best);
      expect(item.miss_distance_cvar_alpha).toBe(0.75);
      expect(item.miss_distance_cvar_m).toBeGreaterThanOrEqual(0);
      expect(item.target_hold_probability).toBeGreaterThanOrEqual(0);
      expect(item.target_hold_probability).toBeLessThanOrEqual(1);
      [item.short_risk, item.long_risk, item.left_risk, item.right_risk]
        .forEach((risk) => {
          expect(risk.probability).toBeGreaterThanOrEqual(0);
          expect(risk.probability).toBeLessThanOrEqual(1);
          expect(risk.mean_excess_m).toBeGreaterThanOrEqual(0);
          expect(risk.conditional_mean_excess_m).toBeGreaterThanOrEqual(0);
        });
    });
    expect(result.outcomes.every((item) => item.perfect_information.cost >= 0)).toBe(true);
    expect(result.outcomes.every((item) => Number.isFinite(item.information_cost_delta)))
      .toBe(true);
    for (let trial = 0; trial < 4; trial += 1) {
      const paired = result.outcomes.filter((item) => item.trial_index === trial);
      expect(new Set(paired.map((item) => JSON.stringify(item.true_wind))).size).toBe(1);
    }
  });

  it("reports nonconvergence without emitting nonfinite scatter points", () => {
    const launch = directLaunch({
      ballSpeedMph: 150,
      launchAngleDeg: 45,
      azimuthDeg: 0,
      spinRpm: 2500,
      spinAxisTiltDeg: 0,
    });
    const result = analyzeWindStrategies({
      uncertainty: spec(2),
      strategies: [{ id: "lofted", label: "Lofted", launch, crosswind_aim_gain_rad_per_mps: 0 }],
      target: { forward_m: 100, right_m: 0 },
      analysis: {
        model_name: "waterloo_penner",
        max_time_s: 0.01,
        time_step_s: 0.001,
        miss_scale_m: 20,
        failure_cost: 37,
        target_radius_m: 5,
        miss_distance_cvar_alpha: 0.5,
      },
    });

    expect(result.outcomes.every((item) => item.status === "nonconverged")).toBe(true);
    expect(result.outcomes.every((item) => item.landing_forward_m === null)).toBe(true);
    expect(result.summaries[0].expected_cost).toBe(37);
  });

  it("retains nonfinite integrations as an invalid failure cohort", () => {
    const launch = {
      ...directLaunch({
        ballSpeedMph: 150,
        launchAngleDeg: 12,
        azimuthDeg: 0,
        spinRpm: 2500,
        spinAxisTiltDeg: 0,
      }),
      ballSpeedMps: Number.MAX_VALUE,
    };
    const result = analyzeWindStrategies({
      uncertainty: spec(1),
      strategies: [{ id: "overflow", label: "Overflow", launch, crosswind_aim_gain_rad_per_mps: 0 }],
      target: { forward_m: 100, right_m: 0 },
      analysis: {
        model_name: "waterloo_penner",
        max_time_s: 0.001,
        time_step_s: 0.001,
        miss_scale_m: 20,
        failure_cost: 29,
        target_radius_m: 5,
        miss_distance_cvar_alpha: 0.5,
      },
    });

    expect(result.outcomes[0].status).toBe("invalid");
    expect(result.outcomes[0].landing_forward_m).toBeNull();
    expect(result.outcomes[0].cost).toBe(29);
  });

  it("uses the same policy with true wind for the perfect-information counterfactual", () => {
    const launch = directLaunch({
      ballSpeedMph: 150,
      launchAngleDeg: 12,
      azimuthDeg: 0,
      spinRpm: 2500,
      spinAxisTiltDeg: 0,
    });
    const exactEstimate: WindUncertaintySpec = {
      schema_version: "wind-uncertainty/v1",
      trials: 3,
      seed: 7,
      true_speed_mps: { kind: "fixed", center: 5, minimum: 0 },
      true_from_bearing_deg: { kind: "fixed", center: 90 },
      estimate_error: {
        speed_bias_mps: 0,
        speed_std_mps: 0,
        bearing_bias_deg: 0,
        bearing_std_deg: 0,
        correlation: 0,
      },
      provenance: "test/exact-estimate",
    };
    const result = analyzeWindStrategies({
      uncertainty: exactEstimate,
      strategies: [{
        id: "compensated",
        label: "Compensated",
        launch,
        crosswind_aim_gain_rad_per_mps: 0.2 * Math.PI / 180,
      }],
      target: { forward_m: 230, right_m: 0 },
      analysis: {
        model_name: "waterloo_penner",
        max_time_s: 10,
        time_step_s: 0.01,
        miss_scale_m: 20,
        failure_cost: 100,
        target_radius_m: 10,
        miss_distance_cvar_alpha: 0.9,
      },
    });

    result.outcomes.forEach((item) => {
      expect(item.cost).toBeCloseTo(item.perfect_information.cost, 12);
      expect(item.information_cost_delta).toBeCloseTo(0, 12);
    });
    expect(result.summaries[0].expected_information_cost_delta).toBeCloseTo(0, 12);
  });

  it("includes failures in target-hold and miss-distance tail risk", () => {
    const failureCase = riskFixture.failure_case;
    const launch = directLaunch({
      ballSpeedMph: 150,
      launchAngleDeg: 45,
      azimuthDeg: 0,
      spinRpm: 2500,
      spinAxisTiltDeg: 0,
    });
    const result = analyzeWindStrategies({
      uncertainty: spec(2),
      strategies: [{ id: "lofted", label: "Lofted", launch, crosswind_aim_gain_rad_per_mps: 0 }],
      target: { forward_m: 100, right_m: 0 },
      analysis: {
        model_name: "waterloo_penner",
        max_time_s: 0.01,
        time_step_s: 0.001,
        miss_scale_m: failureCase.miss_scale_m,
        failure_cost: failureCase.failure_cost,
        target_radius_m: failureCase.target_radius_m,
        miss_distance_cvar_alpha: failureCase.miss_distance_cvar_alpha,
      },
    });

    const summary = result.summaries[0];
    const expected = failureCase.expected;
    expect(result.schema_version).toBe(riskFixture.schema_version);
    expect(summary.target_hold_probability).toBe(expected.target_hold_probability);
    expect(summary.miss_distance_cvar_m).toBe(expected.miss_distance_cvar_m);
    expect(summary.expected_information_cost_delta)
      .toBe(expected.expected_information_cost_delta);
    expect(summary.expected_preset_oracle_regret)
      .toBe(expected.expected_preset_oracle_regret);
    [summary.short_risk, summary.long_risk, summary.left_risk, summary.right_risk]
      .forEach((risk) => expect(risk.probability).toBe(expected.directional_probability));
  });

  it.each([
    { target_radius_m: -0.1, miss_distance_cvar_alpha: 0.9 },
    { target_radius_m: 5, miss_distance_cvar_alpha: 0 },
    { target_radius_m: 5, miss_distance_cvar_alpha: 1 },
  ])("rejects invalid risk configuration %#", (riskConfig) => {
    const launch = directLaunch({
      ballSpeedMph: 150,
      launchAngleDeg: 12,
      azimuthDeg: 0,
      spinRpm: 2500,
      spinAxisTiltDeg: 0,
    });
    expect(() => analyzeWindStrategies({
      uncertainty: spec(1),
      strategies: [{ id: "straight", label: "Straight", launch, crosswind_aim_gain_rad_per_mps: 0 }],
      target: { forward_m: 230, right_m: 0 },
      analysis: {
        model_name: "waterloo_penner",
        max_time_s: 10,
        time_step_s: 0.01,
        miss_scale_m: 20,
        failure_cost: 100,
        ...riskConfig,
      },
    })).toThrow();
  });

  it("matches the cross-language directional-risk golden fixture", () => {
    const golden = riskFixture.directional_case;
    const launch = directLaunch({
      ballSpeedMph: 150, launchAngleDeg: 12, azimuthDeg: 0,
      spinRpm: 2500, spinAxisTiltDeg: 0,
    });
    const request: WindStrategyRequest = {
      uncertainty: spec(golden.landings.length),
      strategies: [{
        id: "fixture", label: "Fixture", launch,
        crosswind_aim_gain_rad_per_mps: 0,
      }],
      target: golden.target,
      analysis: {
        model_name: "waterloo_penner", max_time_s: 10, time_step_s: 0.01,
        miss_scale_m: golden.miss_scale_m, failure_cost: golden.failure_cost,
        target_radius_m: golden.target_radius_m,
        miss_distance_cvar_alpha: golden.miss_distance_cvar_alpha,
      },
    };
    const calm = { ...meteorologicalWind(0, 0), provenance: "test/golden-risk" };
    const outcomes: WindStrategyOutcome[] = golden.landings.map((landing, index) => {
      const status: WindOutcomeStatus = landing === null ? "nonconverged" : "completed";
      const forward = landing?.forward_m ?? null;
      const right = landing?.right_m ?? null;
      const miss = forward === null || right === null ? null :
        Math.hypot(forward - golden.target.forward_m, right - golden.target.right_m);
      const cost = miss === null ? golden.failure_cost : (miss / golden.miss_scale_m) ** 2;
      const failureReason = landing === null ? "fixture failure" : null;
      return {
        trial_index: index, strategy_id: "fixture", status, true_wind: calm,
        estimated_wind: calm, landing_forward_m: forward, landing_right_m: right,
        miss_distance_m: miss, cost, failure_reason: failureReason,
        perfect_information: {
          status, landing_forward_m: forward, landing_right_m: right,
          miss_distance_m: miss, cost, failure_reason: failureReason,
        },
        information_cost_delta: 0,
      };
    });

    const summary = summarizeStrategyOutcomes(request, outcomes)[0];
    const expected = golden.expected;
    expect(summary.expected_cost).toBeCloseTo(expected.expected_cost, 12);
    expect(summary.target_hold_probability).toBe(expected.target_hold_probability);
    expect(summary.miss_distance_cvar_m).toBeCloseTo(expected.miss_distance_cvar_m, 12);
    expect(summary.short_risk).toEqual(expected.short_risk);
    expect(summary.long_risk).toEqual(expected.long_risk);
    expect(summary.left_risk).toEqual(expected.left_risk);
    expect(summary.right_risk).toEqual(expected.right_risk);
  });
});
