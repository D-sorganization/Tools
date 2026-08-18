import { describe, expect, it } from "vitest";

import { simulateFlightWithOptions, type AngularFlightPoint } from "./flight";
import {
  buildFlightToGroundRequest,
  launchRelativeSurfaceHeightM,
  simulateFlightForGroundTransfer,
  type FlightToGroundTransferConfig,
} from "./flightGroundTransfer";
import type { Launch } from "./flight";
import type { GroundSurfaceProfile } from "./flightGroundTypes";

const surface: GroundSurfaceProfile = {
  surface_id: "test-plane",
  provider_id: "tools.planar-surface",
  provider_version: "1.0.0",
  frame: "target_frame:x_downrange,y_up,z_right",
  height_m: 0,
  normal_unit: [0, 1, 0],
  surface_velocity_m_s: [0, 0, 0],
  normal_restitution: 0.42,
  static_friction: 0.35,
  kinetic_friction: 0.28,
  rolling_resistance: 0.04,
  firmness_pa: 1_200_000,
  hardness_fraction: 0.7,
  grass_height_m: 0.012,
  compressibility_fraction: 0.2,
  compression_damping_fraction: 0.25,
  turf_density_kg_m3: 180,
  moisture_fraction: 0.3,
};

const config: FlightToGroundTransferConfig = {
  requestId: "transfer-1",
  flightOrigin: [100, 5, 2],
  launchRelativeSurface: surface,
  surfaceOrigin: "launch_ball_center",
  ballRadiusM: 0.02135,
  ballMassKg: 0.04593,
  rotationalInertiaFactor: 0.4,
  maxTimeS: 12,
  outputIntervalS: 0.01,
  maxEvents: 64,
  calibration: {
    calibration_id: "test-calibration",
    kind: "unvalidated",
    source: "test fixture",
    confidence: 0,
  },
  provenance: {
    producer: "tools.rate_of_closure.web",
    producer_version: "1.0.0",
    source_revision: "test",
    input_sha256: "a".repeat(64),
  },
};

const point = (
  time: number,
  relativePosition: readonly [number, number, number],
  relativeVelocity: readonly [number, number, number],
): AngularFlightPoint => ({
  time,
  position: [
    config.flightOrigin[0] + relativePosition[0],
    config.flightOrigin[1] + relativePosition[1],
    config.flightOrigin[2] + relativePosition[2],
  ],
  velocity: [...relativeVelocity],
  angularVelocityRadS: [4, 2, 260],
});

const withOrigin = (...points: AngularFlightPoint[]): AngularFlightPoint[] => [
  point(0, [0, 0, 0], [10, 0, 3]),
  ...points,
];

describe("flight-to-ground transfer", () => {
  it("converts canonical flight coordinates and brackets physical sphere contact", () => {
    const outcome = buildFlightToGroundRequest(withOrigin(
      point(5.19, [209.7, 3.01, 0.024], [31, -1.5, -12]),
      point(5.2, [210, 3, 0.019], [31, -1.5, -12]),
    ), config);

    expect(outcome.status).toBe("available");
    if (outcome.status !== "available") return;
    expect(outcome.request.last_separated_state).toEqual({
      time_s: 5.19,
      frame: "target_frame:x_downrange,y_up,z_right",
      position_m: [209.7, 0.024, -3.01],
      velocity_m_s: [31, -12, 1.5],
      angular_velocity_rad_s: [4, 260, -2],
    });
    expect(outcome.request.first_penetrating_state.position_m).toEqual([210, 0.019, -3]);
  });

  it("reports no-crossing and grazing outcomes without fabricating a request", () => {
    const noCrossing = buildFlightToGroundRequest(withOrigin(
      point(0.1, [0, 0, 1], [10, 0, -1]),
      point(1, [10, 0, 0.5], [10, 0, -1]),
    ), config);
    const grazing = buildFlightToGroundRequest(withOrigin(
      point(0.1, [0, 0, 0.03], [10, 0, -0.02]),
      point(1, [10, 0, 0.02135], [10, 0, 0]),
      point(2, [20, 0, 0.03], [10, 0, 0.02]),
    ), config);

    expect(noCrossing).toMatchObject({ status: "unavailable", reason: "no_physical_contact" });
    expect(grazing).toMatchObject({ status: "unavailable", reason: "grazing_contact" });
  });

  it("supports arbitrary upward plane normals and tangential surface motion", () => {
    const component = Math.SQRT1_2;
    const slopedConfig: FlightToGroundTransferConfig = {
      ...config,
      ballRadiusM: 0.02,
      launchRelativeSurface: {
        ...surface,
        normal_unit: [0, component, component],
        surface_velocity_m_s: [1, 0, 0],
      },
    };
    const incomingFlightVelocity: readonly [number, number, number] = [1, 2 * component, -2 * component];
    const outcome = buildFlightToGroundRequest(withOrigin(
      point(1, [4, -(0.023 * component), 0.023 * component], incomingFlightVelocity),
      point(1.01, [4.01, -(0.019 * component), 0.019 * component], incomingFlightVelocity),
    ), slopedConfig);

    expect(outcome.status).toBe("available");
  });

  it("reports missing signed angular state explicitly", () => {
    const missingAngular = point(1, [0, 0, 0.03], [10, 0, -2]) as Partial<AngularFlightPoint>;
    delete missingAngular.angularVelocityRadS;
    const outcome = buildFlightToGroundRequest(withOrigin(
      missingAngular as AngularFlightPoint,
      point(1.1, [1, 0, 0.019], [10, 0, -2]),
    ), config);

    expect(outcome).toMatchObject({
      status: "unavailable",
      reason: "missing_terminal_angular_velocity",
    });
  });

  it("does not immediately retrigger a zero-time contact after a bounce", () => {
    const outcome = buildFlightToGroundRequest(withOrigin(
      point(2, [0, 0, 0.02135], [12, 0, 3]),
      point(2.1, [1, 0, 0.08], [11, 0, 2]),
      point(2.2, [2, 0, 0.03], [10, 0, -2]),
      point(2.3, [3, 0, 0.019], [9, 0, -2]),
    ), config);

    expect(outcome.status).toBe("available");
    if (outcome.status !== "available") return;
    expect(outcome.request.last_separated_state.time_s).toBe(2.2);
    expect(outcome.request.first_penetrating_state.time_s).toBe(2.3);
  });

  it("does not classify a launch contact with no later return as grazing", () => {
    const groundConfig: FlightToGroundTransferConfig = {
      ...config,
      launchRelativeSurface: { ...surface, height_m: -config.ballRadiusM },
    };
    const outcome = buildFlightToGroundRequest([
      point(0, [0, 0, 0], [10, 0, 3]),
      point(0.1, [1, 0, 0.1], [10, 0, 2]),
    ], groundConfig);

    expect(outcome).toMatchObject({ status: "unavailable", reason: "no_physical_contact" });
  });

  it("rejects a missing or mismatched launch-origin datum", () => {
    const bracket = [
      point(1, [1, 0, 0.03], [10, 0, -2]),
      point(1.1, [2, 0, 0.019], [10, 0, -2]),
    ];
    const wrongOrigin: FlightToGroundTransferConfig = {
      ...config,
      flightOrigin: [config.flightOrigin[0] + 1, config.flightOrigin[1], config.flightOrigin[2]],
    };

    expect(buildFlightToGroundRequest(bracket, config)).toMatchObject({
      status: "unavailable",
      reason: "invalid_trajectory",
    });
    expect(buildFlightToGroundRequest(withOrigin(...bracket), wrongOrigin)).toMatchObject({
      status: "unavailable",
      reason: "invalid_trajectory",
    });
  });

  it("expresses ground and tee planes relative to the launch-centre origin", () => {
    expect(launchRelativeSurfaceHeightM(0.02135, {
      supportMode: "ground",
      teeHeightM: 0,
    })).toBe(-0.02135);
    expect(launchRelativeSurfaceHeightM(0.02135, {
      supportMode: "tee",
      teeHeightM: 0.0381,
    })).toBe(-0.05945);
  });

  it("extends a teed flight to the physical sphere/plane crossing", () => {
    const launch: Launch = {
      ballSpeedMps: 35,
      launchAngleRad: 0.3,
      azimuthRad: 0.02,
      spinRpm: 2500,
      spinAxis: [0, -1, 0],
    };
    const teeConfig: FlightToGroundTransferConfig = {
      ...config,
      flightOrigin: [0, 0, 0],
      launchRelativeSurface: {
        ...surface,
        height_m: launchRelativeSurfaceHeightM(0.02135, {
          supportMode: "tee",
          teeHeightM: 0.0381,
        }),
      },
    };
    const flight = simulateFlightForGroundTransfer(launch, teeConfig, {
      maxTimeS: 10,
      stepS: 0.002,
      sampleEvery: 5,
    });

    const outcome = buildFlightToGroundRequest(flight.trajectory, teeConfig);
    expect(outcome.status).toBe("available");
    if (outcome.status !== "available") return;
    expect(outcome.request.last_separated_state.position_m[1]).toBeGreaterThan(-0.0381);
    expect(outcome.request.first_penetrating_state.position_m[1]).toBeCloseTo(-0.0381, 8);
  });

  it("rejects signed spin magnitude because sign belongs in the axis", () => {
    const teeConfig: FlightToGroundTransferConfig = {
      ...config,
      flightOrigin: [0, 0, 0],
      launchRelativeSurface: { ...surface, height_m: -0.05945 },
    };
    expect(() => simulateFlightForGroundTransfer({
      ballSpeedMps: 35,
      launchAngleRad: 0.3,
      azimuthRad: 0,
      spinRpm: -2500,
      spinAxis: [0, -1, 0],
    }, teeConfig)).toThrow(/spinRpm.*nonnegative/);
  });

  it("rejects configurations that exceed the synchronous RK4 step budget", () => {
    const teeConfig: FlightToGroundTransferConfig = {
      ...config,
      flightOrigin: [0, 0, 0],
      launchRelativeSurface: { ...surface, height_m: -0.05945 },
    };
    expect(() => simulateFlightForGroundTransfer({
      ballSpeedMps: 35,
      launchAngleRad: 0.3,
      azimuthRad: 0,
      spinRpm: 2500,
      spinAxis: [0, -1, 0],
    }, teeConfig, {
      maxTimeS: 1000,
      stepS: 0.0005,
    })).toThrow(/integration step budget.*50,000/);
  });

  it("accepts exactly 50,000 synchronous RK4 steps", () => {
    const result = simulateFlightWithOptions({
      ballSpeedMps: 1,
      launchAngleRad: 0.1,
      azimuthRad: 0,
      spinRpm: 0,
      spinAxis: [0, 1, 0],
    }, {
      maxTimeS: 50,
      stepS: 0.001,
      sampleEvery: 50_000,
      terminalGapM: () => 1,
    });

    expect(result.flightTimeS).toBeCloseTo(50, 12);
    expect(result.trajectory).toHaveLength(2);
  });

  it("uses an exact partial final step without exceeding maxTimeS", () => {
    const result = simulateFlightWithOptions({
      ballSpeedMps: 1,
      launchAngleRad: 0.1,
      azimuthRad: 0,
      spinRpm: 0,
      spinAxis: [0, 1, 0],
    }, {
      maxTimeS: 0.015,
      stepS: 0.01,
      sampleEvery: 1,
      terminalGapM: () => 1,
    });

    expect(result.flightTimeS).toBeCloseTo(0.015, 14);
    expect(result.trajectory[result.trajectory.length - 1]?.time).toBeCloseTo(0.015, 14);
  });

  it("accepts exact sphere contact as the first penetrating bracket state", () => {
    const outcome = buildFlightToGroundRequest(withOrigin(
      point(0.1, [1, 0, 0.03], [10, 0, -2]),
      point(0.2, [2, 0, 0.02135], [10, 0, -2]),
    ), config);

    expect(outcome.status).toBe("available");
    if (outcome.status !== "available") return;
    expect(outcome.request.first_penetrating_state.position_m[1]).toBe(0.02135);
  });

  it("stops at exact tee contact even when it occurs in the first RK4 step", () => {
    const teeConfig: FlightToGroundTransferConfig = {
      ...config,
      flightOrigin: [0, 0, 0],
      launchRelativeSurface: { ...surface, height_m: -0.05945 },
    };
    const flight = simulateFlightForGroundTransfer({
      ballSpeedMps: 10,
      launchAngleRad: -Math.PI / 2,
      azimuthRad: 0,
      spinRpm: 0,
      spinAxis: [0, 1, 0],
    }, teeConfig, { maxTimeS: 0.01, stepS: 0.01, sampleEvery: 1 });
    const outcome = buildFlightToGroundRequest(flight.trajectory, teeConfig);

    expect(flight.flightTimeS).toBeLessThan(0.01);
    expect(outcome.status).toBe("available");
    if (outcome.status !== "available") return;
    expect(outcome.request.first_penetrating_state.position_m[1]).toBeCloseTo(-0.0381, 12);
  });
});
