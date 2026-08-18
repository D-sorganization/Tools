import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/ball_flight_metrics_golden_v1.json";
import {
  FLIGHT_METRIC_IDS,
  flightMetricCatalog,
  parseFlightMetricCatalog,
  stableFlightMetricCatalogJson,
} from "./ballFlightMetricContract";
import {
  deriveFlightMetricResult,
  stableFlightMetricResultJson,
  type FlightMetricInputs,
  type FlightRunManifest,
} from "./ballFlightMetrics";
import { cross, deliveryDiagnostics, type DeliveryInput } from "./impactPhysics";
import { spinAxisTiltDeg } from "./spinAxisConvention";

const manifest: FlightRunManifest = {
  modelId: "analytic_fixture", modelVersion: "1.0.0",
  integrationStatus: "complete", terminationReason: "ground_crossing",
  environment: { air_density_kg_m3: "1.225", gravity_m_s2: "9.80665" },
  wind: { model: "still_air" }, uncertaintyStatus: "deterministic",
  frameId: "target_frame:x_downrange,y_up,z_right",
};

const analyticInputs = (): FlightMetricInputs => ({
  trajectory: fixture.analytic_case.trajectory.map((point) => ({
    timeS: point.time_s,
    positionM: point.position_m as [number, number, number],
    velocityMps: point.velocity_m_s as [number, number, number],
  })),
  spinVectorRpm: fixture.analytic_case.spin_vector_rpm as [number, number, number],
  targetPositionM: fixture.analytic_case.target_position_m as [number, number, number],
});

describe("ball-flight result contract", () => {
  it("is complete, strict, deterministic, and Python-identical", async () => {
    const catalog = flightMetricCatalog();
    expect(catalog.definitions.map((item) => item.metricId).sort())
      .toEqual([...FLIGHT_METRIC_IDS].sort());
    expect(catalog.definitions.every((item) => item.coverage.length === 3)).toBe(true);
    const serialized = stableFlightMetricCatalogJson(catalog);
    expect(stableFlightMetricCatalogJson(parseFlightMetricCatalog(JSON.parse(serialized))))
      .toBe(serialized);
    expect(() => parseFlightMetricCatalog({ ...JSON.parse(serialized), extra: true }))
      .toThrow(/catalog fields/);
    const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(serialized));
    expect(Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join(""))
      .toBe(fixture.catalog_sha256);
  });

  it("interpolates the analytic landing and matches Python scalars", () => {
    const result = deriveFlightMetricResult(analyticInputs(), manifest);
    for (const [metricId, expected] of Object.entries(fixture.analytic_case.expected_scalars)) {
      expect(result.scalar(metricId)).toBeCloseTo(expected, 10);
    }
    expect(result.vector("landing_position")).toEqual([25, 0, 4]);
    expect(result.value("total_distance")).toMatchObject({
      status: "unavailable", reason: "ground_model_required", numeric: null,
    });
  });

  it("serializes the complete analytic result identically to Python", async () => {
    const serialized = stableFlightMetricResultJson(
      deriveFlightMetricResult(analyticInputs(), manifest),
    );
    const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(serialized));
    expect(Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join(""))
      .toBe(fixture.analytic_result_sha256);
  });

  it("rejects unordered trajectories and preserves typed unavailable reasons", () => {
    expect(() => deriveFlightMetricResult({
      trajectory: [
        { timeS: 1, positionM: [0, 1, 0], velocityMps: [1, 0, 0] },
        { timeS: 1, positionM: [1, 0, 0], velocityMps: [1, -1, 0] },
      ],
      spinVectorRpm: [0, 0, 0],
    }, manifest)).toThrow(/strictly increasing/);
    const result = deriveFlightMetricResult({
      trajectory: [{ timeS: 0, positionM: [0, 0, 0], velocityMps: [0, 0, 0] }],
      spinVectorRpm: [0, 0, 0],
    }, manifest);
    expect(result.value("spin_axis_tilt").reason).toBe("zero_spin");
    expect(result.value("carry_distance").reason).toBe("insufficient_trajectory");
    expect(result.value("target_residual").reason).toBe("target_not_configured");
  });

  it("accepts ground values only with an identified qualified model", () => {
    const result = deriveFlightMetricResult({
      ...analyticInputs(),
      groundResult: {
        modelId: "qualified-ground/v1", totalDistanceM: 28,
        rollDistanceM: 2.7, bounceCount: 2, finalOfflineM: 4.5,
      },
    }, manifest);
    expect(result.scalar("total_distance")).toBe(28);
    expect(result.value("total_distance")).toMatchObject({
      status: "model_dependent", provenance: "qualified-ground/v1",
    });
    expect(() => deriveFlightMetricResult({
      ...analyticInputs(),
      groundResult: {
        modelId: "invalid", totalDistanceM: 28,
        rollDistanceM: 2.7, bounceCount: 1.5, finalOfflineM: 4.5,
      },
    }, manifest)).toThrow(/bounceCount must be an integer/);
  });

  it("uses positive spin-axis tilt for fade/right curvature", () => {
    const tilt = 10 * Math.PI / 180;
    const result = deriveFlightMetricResult({
      ...analyticInputs(),
      spinVectorRpm: [0, -1000 * Math.sin(tilt), 1000 * Math.cos(tilt)],
    }, manifest);

    expect(result.scalar("spin_axis_tilt")).toBeCloseTo(10, 10);
    expect(flightMetricCatalog().definition("spin_axis_tilt").signRule)
      .toBe("positive_right");
  });

  it("projects gyro spin out consistently across result and impact producers", () => {
    const spin: [number, number, number] = [500, -100, 1000];
    const expected = spinAxisTiltDeg(spin);
    const result = deriveFlightMetricResult({
      ...analyticInputs(), spinVectorRpm: spin,
    }, manifest);
    const input: DeliveryInput = {
      clubheadSpeedMps: 40, clubPathDeg: 12, faceAngleDeg: -4,
      attackAngleDeg: -5, dynamicLoftDeg: 23,
      impactOffsetToeMm: 0, impactOffsetHighMm: 0,
    };
    const diagnostics = deliveryDiagnostics(input);
    const radians = (degrees: number) => degrees * Math.PI / 180;
    const velocity = [
      Math.cos(radians(input.attackAngleDeg)) * Math.cos(radians(input.clubPathDeg)),
      Math.sin(radians(input.attackAngleDeg)),
      Math.cos(radians(input.attackAngleDeg)) * Math.sin(radians(input.clubPathDeg)),
    ] as [number, number, number];
    const normal = [
      Math.cos(radians(input.dynamicLoftDeg)) * Math.cos(radians(input.faceAngleDeg)),
      Math.sin(radians(input.dynamicLoftDeg)),
      Math.cos(radians(input.dynamicLoftDeg)) * Math.sin(radians(input.faceAngleDeg)),
    ] as [number, number, number];
    const impactExpected = spinAxisTiltDeg(cross(velocity, normal));

    expect(expected).toBeCloseTo(Math.atan2(100, 1000) * 180 / Math.PI, 10);
    expect(result.scalar("spin_axis_tilt")).toBeCloseTo(expected as number, 10);
    expect(diagnostics.spinAxisTiltDeg).toBeCloseTo(impactExpected as number, 10);
  });
});
