/** Cross-runtime parity against the Python-generated Morris UI fixture. */

import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/morris_ui_parity_v1.json";
import oracleFixture from "./__fixtures__/morris_analytical_ground_truth_v1.json";
import { CLUB_LIBRARY } from "./club";
import { parseMorrisJobEnvelope } from "./morrisAuthorityContract";
import { validateMorrisMetrics } from "./morrisMetricValidation";
import {
  buildMorrisFactorRows,
  AUTHORITY_FLIGHT_MODELS,
  serializeMorrisAuthorityRequest,
  type MorrisAuthorityBase,
  type MorrisAuthorityRequest,
  type MorrisFactorDraft,
} from "./morrisAuthorityRequest";
import { presentMorrisJob, presentMorrisReport } from "./morrisPresentation";

const snakeCase = (key: string): string =>
  key.replace(/[A-Z]/g, (character) => `_${character.toLowerCase()}`);

const snakeDocument = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(snakeDocument);
  if (value === null || typeof value !== "object") return value;
  return Object.fromEntries(
    Object.entries(value).map(([key, entry]) => [
      snakeCase(key),
      snakeDocument(entry),
    ]),
  );
};

const drafts = (): MorrisFactorDraft[] =>
  fixture.factor_drafts.map((draft) => ({
    variableKey: draft.variable_key,
    enabled: draft.enabled,
    lower: draft.lower,
    upper: draft.upper,
  }));

const base = (): MorrisAuthorityBase => {
  const source = fixture.submitted_request.base;
  return {
    clubName: source.club_name,
    supportMode: source.support_mode as MorrisAuthorityBase["supportMode"],
    teeHeightM: source.tee_height_m,
    planeYawDeg: source.plane_yaw_deg,
    planeSideTiltDeg: source.plane_side_tilt_deg,
    planeForwardTiltDeg: source.plane_forward_tilt_deg,
    pendulumM1Kg: source.pendulum_m1_kg,
    pendulumL1M: source.pendulum_l1_m,
    pendulumLc1M: source.pendulum_lc1_m,
    pendulumI1KgM2: source.pendulum_i1_kg_m2,
    pendulumM2Kg: source.pendulum_m2_kg,
    pendulumL2M: source.pendulum_l2_m,
    pendulumLc2M: source.pendulum_lc2_m,
    pendulumI2KgM2: source.pendulum_i2_kg_m2,
    dampingShoulder: source.damping_shoulder,
    dampingWrist: source.damping_wrist,
    swingDurationS: source.swing_duration_s,
    flightModel: source.flight_model,
    impactOffsetToeMm: source.impact_offset_toe_mm,
    impactOffsetHighMm: source.impact_offset_high_mm,
  };
};

const request = (): MorrisAuthorityRequest => ({
  requestId: fixture.submitted_request.request_id,
  base: base(),
  factors: drafts(),
  trajectories: fixture.submitted_request.trajectories,
  levels: fixture.submitted_request.levels,
  seed: fixture.submitted_request.seed,
  minimumEffects: fixture.submitted_request.minimum_effects,
  workerCount: fixture.submitted_request.worker_count,
});

describe("Python-generated Morris UI parity", () => {
  it("matches factor rows and the exact submitted request", () => {
    expect(fixture.schema_id).toBe("rate-of-closure/morris-ui-parity");
    expect(fixture.schema_version).toBe(1);
    expect(CLUB_LIBRARY.map((club) => club.name)).toEqual(
      fixture.authority_club_names,
    );
    expect(AUTHORITY_FLIGHT_MODELS).toEqual(fixture.authority_flight_models);

    const rows = buildMorrisFactorRows(drafts(), "tee");

    expect(snakeDocument(rows)).toEqual(fixture.expected_factor_rows);
    expect(serializeMorrisAuthorityRequest(request())).toEqual(
      fixture.submitted_request,
    );
  });

  it("matches completed lifecycle and every target-scoped table", () => {
    const job = parseMorrisJobEnvelope(structuredClone(fixture.completed_job));

    expect(snakeDocument(presentMorrisJob(job))).toEqual(
      fixture.expected_job_presentation,
    );
    for (const [targetName, expected] of Object.entries(
      fixture.expected_tables,
    )) {
      expect(
        snakeDocument(presentMorrisReport(job.report!, targetName)),
      ).toEqual(expected);
    }
  });
});

interface AnalyticalOracleResult {
  readonly mu: number;
  readonly muStar: number;
  readonly sigma: number;
  readonly standardError: number;
}

const morrisLinearAnalyticalOracle = (
  coefficient: number,
  lower: number,
  upper: number,
): AnalyticalOracleResult => {
  if (lower >= upper)
    throw new RangeError("lower must be strictly less than upper");
  const span = upper - lower;
  return {
    mu: coefficient * span,
    muStar: Math.abs(coefficient) * span,
    sigma: 0,
    standardError: 0,
  };
};

const morrisPolynomialQuadraticAnalyticalOracle = (
  coefficient: number,
  validPairs: number,
): AnalyticalOracleResult => {
  if (validPairs < 2) throw new RangeError("validPairs must be at least 2");
  const mu = coefficient;
  const muStar = Math.abs(coefficient);
  const variance = (validPairs / (validPairs - 1)) * (coefficient / 3) ** 2;
  const sigma = Math.sqrt(variance);
  const standardError = sigma / Math.sqrt(validPairs);
  return { mu, muStar, sigma, standardError };
};

describe("Morris closed-form analytical test oracle and ground truth", () => {
  it("verifies analytical oracle derivations match expected ground truth fixture", () => {
    expect(oracleFixture.schema_id).toBe(
      "rate-of-closure/morris-analytical-ground-truth",
    );
    expect(oracleFixture.schema_version).toBe(1);

    const sideTilt = morrisLinearAnalyticalOracle(-3.5, -15.0, 5.0);
    expect(sideTilt.mu).toBe(-70.0);
    expect(sideTilt.muStar).toBe(70.0);
    expect(sideTilt.sigma).toBe(0.0);
    expect(sideTilt.standardError).toBe(0.0);

    const yaw = morrisLinearAnalyticalOracle(2.5, -2.0, 8.0);
    expect(yaw.mu).toBe(25.0);
    expect(yaw.muStar).toBe(25.0);
    expect(yaw.sigma).toBe(0.0);
    expect(yaw.standardError).toBe(0.0);

    const shoulderDamping = morrisPolynomialQuadraticAnalyticalOracle(15.0, 12);
    expect(shoulderDamping.mu).toBe(15.0);
    expect(shoulderDamping.muStar).toBe(15.0);
    expect(shoulderDamping.sigma).toBeCloseTo(
      (10 * Math.sqrt(3)) / Math.sqrt(11),
      12,
    );
    expect(shoulderDamping.standardError).toBeCloseTo(5 / Math.sqrt(11), 12);

    for (const factor of oracleFixture.analytical_factors) {
      if (factor.model_type === "linear") {
        const derived = morrisLinearAnalyticalOracle(
          factor.coefficient,
          factor.lower,
          factor.upper,
        );
        expect(derived.mu).toBeCloseTo(factor.expected_mu, 12);
        expect(derived.muStar).toBeCloseTo(factor.expected_mu_star, 12);
        expect(derived.sigma).toBeCloseTo(factor.expected_sigma, 12);
        expect(derived.standardError).toBeCloseTo(
          factor.expected_standard_error,
          12,
        );
      } else if (factor.model_type === "polynomial_quadratic") {
        const derived = morrisPolynomialQuadraticAnalyticalOracle(
          factor.coefficient,
          12,
        );
        expect(derived.mu).toBeCloseTo(factor.expected_mu, 12);
        expect(derived.muStar).toBeCloseTo(factor.expected_mu_star, 12);
        expect(derived.sigma).toBeCloseTo(factor.expected_sigma, 12);
        expect(derived.standardError).toBeCloseTo(
          factor.expected_standard_error,
          12,
        );
      } else {
        expect(factor.expected_mu).toBe(0.0);
        expect(factor.expected_mu_star).toBe(0.0);
        expect(factor.expected_sigma).toBe(0.0);
        expect(factor.expected_standard_error).toBe(0.0);
      }
    }
  });

  it("presents analytical ground truth matching exact rankings and scale-aware metrics", () => {
    const job = parseMorrisJobEnvelope(
      structuredClone(oracleFixture.completed_job),
    );
    expect(job.report).not.toBeNull();

    const presentation = presentMorrisReport(job.report!, "clubhead_x_m");
    expect(snakeDocument(presentation)).toEqual(
      oracleFixture.expected_presentation_table,
    );

    expect(
      presentation.rows.map((row) => [row.rank, row.specId, row.muStar]),
    ).toEqual([
      [1, "swing-side-tilt", 70.0],
      [2, "swing-yaw", 25.0],
      [3, "shoulder-damping", 15.0],
      [4, "wrist-damping", 0.0],
    ]);

    const bareCoefficients = [3.5, 2.5, 15.0, 0.0];
    expect(presentation.rows[0].muStar).not.toBe(bareCoefficients[0]);
    expect(presentation.rows[1].muStar).not.toBe(bareCoefficients[1]);

    const unnormalizedSteps = [70.0 * (2 / 3), 25.0 * (2 / 3)];
    expect(presentation.rows[0].muStar).not.toBeCloseTo(
      unnormalizedSteps[0],
      6,
    );
    expect(presentation.rows[1].muStar).not.toBeCloseTo(
      unnormalizedSteps[1],
      6,
    );

    expect(presentation.rows[0].mu).toBe(-70.0);
    expect(presentation.rows[0].muStar).toBe(70.0);
  });

  it("satisfies metric validation invariant across all analytical oracle factors", () => {
    const job = parseMorrisJobEnvelope(
      structuredClone(oracleFixture.completed_job),
    );
    for (const estimate of job.report!.estimates) {
      expect(() =>
        validateMorrisMetrics(
          estimate.effects,
          estimate.availability,
          estimate.denominator.validPairs,
        ),
      ).not.toThrow();
    }
  });
});
