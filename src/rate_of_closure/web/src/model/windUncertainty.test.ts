import fixture from "./__fixtures__/wind_uncertainty_golden_v1.json";
import { describe, expect, it } from "vitest";

import { directLaunch } from "./flightExplorer";
import {
  analyzeWindStrategies,
  sampleWindTrials,
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
      },
    });

    expect(result.outcomes).toHaveLength(8);
    expect(result.summaries).toHaveLength(2);
    expect(result.summaries.every((item) => item.completed_trials === 4)).toBe(true);
    expect(result.summaries.every((item) => item.expected_regret >= 0)).toBe(true);
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
      },
    });

    expect(result.outcomes[0].status).toBe("invalid");
    expect(result.outcomes[0].landing_forward_m).toBeNull();
    expect(result.outcomes[0].cost).toBe(29);
  });
});
