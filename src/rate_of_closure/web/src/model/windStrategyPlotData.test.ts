import { describe, expect, it, vi } from "vitest";

import { simulateFlight } from "./flight";
import { buildScalarEnsembleScatter } from "./scalarEnsembleContract";
import {
  WIND_STRATEGY_ANALYSIS_SCHEMA_VERSION,
  WIND_UNCERTAINTY_SCHEMA_VERSION,
  type WindOutcomeStatus,
  type WindStrategyAnalysis,
  type WindStrategyRequest,
} from "./windUncertainty";
import { buildWindStrategyPlotData } from "./windStrategyPlotData";

vi.mock("./flight", () => ({ simulateFlight: vi.fn() }));

const strategy = {
  id: "stock",
  label: "Stock wedge",
  launch: {
    ballSpeedMps: 22,
    launchAngleRad: 0.55,
    azimuthRad: 0.02,
    spinRpm: 6200,
    spinAxis: [0, -1, 0] as [number, number, number],
  },
  crosswind_aim_gain_rad_per_mps: 0.01,
};

const request = (): WindStrategyRequest => ({
  uncertainty: {
    schema_version: WIND_UNCERTAINTY_SCHEMA_VERSION,
    trials: 3,
    seed: 7,
    true_speed_mps: { kind: "fixed", center: 5, spread: 0, minimum: 0 },
    true_from_bearing_deg: { kind: "fixed", center: 90, spread: 0 },
    estimate_error: {
      speed_bias_mps: 1, speed_std_mps: 0, bearing_bias_deg: 10,
      bearing_std_deg: 0, correlation: 0,
    },
    provenance: "declared-wind-campaign",
  },
  strategies: [strategy],
  target: { forward_m: 27.432, right_m: 1.5 },
  analysis: {
    model_name: "waterloo_penner", max_time_s: 10, time_step_s: 0.01,
    miss_scale_m: 5, failure_cost: 100, target_radius_m: 3,
    miss_distance_cvar_alpha: 0.9,
  },
});

const scenario = (speed: number, bearingDeg: number, provenance: string) => {
  const bearing = bearingDeg * Math.PI / 180;
  return {
    schemaVersion: "wind-scenario/v1" as const,
    baseVelocityMps: [-speed * Math.cos(bearing), speed * Math.sin(bearing), 0] as [number, number, number],
    shearFractionPer10m: 0,
    gusts: [],
    turbulenceIntensityMps: 0,
    seed: 0,
    provenance,
  };
};

const outcome = (trialIndex: number, status: WindOutcomeStatus) => {
  const completed = status === "completed";
  return {
    trial_index: trialIndex,
    strategy_id: "stock",
    status,
    true_wind: scenario(5, 90, `declared-wind-campaign/true/trial-${trialIndex}`),
    estimated_wind: scenario(6, 100, `declared-wind-campaign/estimated/trial-${trialIndex}`),
    landing_forward_m: completed ? 28 : null,
    landing_right_m: completed ? 2 : null,
    miss_distance_m: completed ? 0.756 : null,
    cost: completed ? 0.02286144 : 100,
    failure_reason: completed ? null : `${status} source result`,
    perfect_information: {
      status,
      landing_forward_m: completed ? 27.7 : null,
      landing_right_m: completed ? 1.7 : null,
      miss_distance_m: completed ? 0.331 : null,
      cost: completed ? 0.00438244 : 100,
      failure_reason: completed ? null : `${status} perfect result`,
    },
    information_cost_delta: completed ? 0.018479 : 0,
  };
};

const analysis = (): WindStrategyAnalysis => ({
  schema_version: WIND_STRATEGY_ANALYSIS_SCHEMA_VERSION,
  provenance: "declared-wind-campaign",
  target: { forward_m: 27.432, right_m: 1.5 },
  wind_trials: [0, 1, 2].map((trialIndex) => ({
    trial_index: trialIndex,
    true_speed_mps: 5,
    true_from_bearing_deg: 90,
    estimated_speed_mps: 6,
    estimated_from_bearing_deg: 100,
    speed_error_mps: 1,
    bearing_error_deg: 10,
  })),
  outcomes: [
    outcome(0, "completed"),
    {
      ...outcome(1, "nonconverged"),
      perfect_information: {
        status: "completed",
        landing_forward_m: 27.6,
        landing_right_m: 1.6,
        miss_distance_m: 0.19,
        cost: 0.001444,
        failure_reason: null,
      },
      information_cost_delta: 99.998556,
    },
    outcome(2, "invalid"),
  ],
  summaries: [],
});

describe("wind strategy plot adapter", () => {
  it("exposes every requested scalar and retains all outcome cohorts", () => {
    const result = buildWindStrategyPlotData(request(), analysis());

    expect(result.result_id).toBe("wind-strategy:declared-wind-campaign");
    expect(result.rows).toHaveLength(3);
    expect(result.rows.map(({ cohort }) => cohort)).toEqual([
      "completed", "nonconverged", "invalid",
    ]);
    expect(result.rows[0]).toMatchObject({
      row_id: "series:stock/trial:0",
      trial_index: 0,
      series_id: "stock",
      values: {
        true_wind_speed_mps: 5,
        estimated_wind_speed_mps: 6,
        wind_speed_error_mps: 1,
        launch_ball_speed_mps: 22,
        launch_angle_rad: 0.55,
        launch_azimuth_rad: 0.02,
        launch_spin_rpm: 6200,
        actual_aim_azimuth_rad: -0.03908846518073247,
        perfect_information_aim_azimuth_rad: -0.030000000000000002,
        target_forward_m: 27.432,
        actual_landing_forward_m: 28,
        perfect_information_landing_forward_m: 27.7,
        information_cost_delta: 0.018479,
      },
      attributes: {
        actual_status: "completed",
        perfect_information_status: "completed",
        strategy_label: "Stock wedge",
      },
    });
    expect(result.rows[1].values.actual_landing_forward_m).toBeNull();
    expect(result.rows[1].values.perfect_information_landing_forward_m).toBe(27.6);
    expect(result.rows[1].attributes?.perfect_information_status).toBe("completed");
    expect(result.rows[2].attributes?.actual_status).toBe("invalid");
  });

  it("provides paired-finite scatter availability without dropping failed rows", () => {
    const result = buildWindStrategyPlotData(request(), analysis());
    const scatter = buildScalarEnsembleScatter(
      result, "actual_landing_forward_m", "actual_landing_right_m",
    );

    expect(scatter.points).toHaveLength(1);
    expect(scatter.availability.by_cohort).toEqual({
      completed: { total_rows: 1, x_finite: 1, y_finite: 1, paired_finite: 1, unavailable: 0 },
      nonconverged: { total_rows: 1, x_finite: 0, y_finite: 0, paired_finite: 0, unavailable: 1 },
      invalid: { total_rows: 1, x_finite: 0, y_finite: 0, paired_finite: 0, unavailable: 1 },
    });
  });

  it("canonicalizes rows by trial and declared strategy order", () => {
    const source = analysis();
    const reversed = { ...source, outcomes: [...source.outcomes].reverse() };

    expect(buildWindStrategyPlotData(request(), reversed).rows.map(({ row_id }) => row_id))
      .toEqual([
        "series:stock/trial:0", "series:stock/trial:1", "series:stock/trial:2",
      ]);
  });

  it("fails closed when request and analysis do not agree", () => {
    const source = analysis();
    const wrongTarget = { ...source, target: { ...source.target, right_m: 9 } };
    expect(() => buildWindStrategyPlotData(request(), wrongTarget)).toThrow(/target.*agree/);

    const missingOutcome = { ...source, outcomes: source.outcomes.slice(1) };
    expect(() => buildWindStrategyPlotData(request(), missingOutcome)).toThrow(/outcome coverage/);

    const first = source.outcomes[0];
    const inconsistentWind = {
      ...source,
      outcomes: [{
        ...first,
        true_wind: { ...first.true_wind, baseVelocityMps: [0, 99, 0] as [number, number, number] },
      }, ...source.outcomes.slice(1)],
    };
    expect(() => buildWindStrategyPlotData(request(), inconsistentWind)).toThrow(/true wind.*agree/);

    const wrongProvenance = {
      ...source,
      outcomes: [{
        ...first,
        true_wind: { ...first.true_wind, provenance: "unrelated-source" },
      }, ...source.outcomes.slice(1)],
    };
    expect(() => buildWindStrategyPlotData(request(), wrongProvenance))
      .toThrow(/deterministic scenario contract/);

    const wrongSamplingRequest = request();
    const changedSampling = {
      ...wrongSamplingRequest,
      uncertainty: {
        ...wrongSamplingRequest.uncertainty,
        true_speed_mps: { ...wrongSamplingRequest.uncertainty.true_speed_mps, center: 8 },
      },
    };
    expect(() => buildWindStrategyPlotData(changedSampling, source)).toThrow(/sampling contract/);
  });

  it("adapts existing results without rerunning flight physics", () => {
    buildWindStrategyPlotData(request(), analysis());

    expect(simulateFlight).not.toHaveBeenCalled();
  });
});
