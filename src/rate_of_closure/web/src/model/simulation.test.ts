/**
 * Parity pins for the simulation port against the canonical Python
 * implementation (pytest: tests/rate_of_closure/test_simulation.py and the
 * pin-generation snippet recorded in the PR). Tight tolerances where the
 * TS code is a formula-for-formula port (pendulum RK4, impact, launch);
 * banded for flight, where scipy RK45 vs fixed-step RK4 differ only by
 * integration error.
 */

import { describe, expect, it } from "vitest";

import {
  BALL_POSITION,
  DEFAULT_IMPACT_CLUB,
  deriveLaunch,
  golfDefaultParams,
  inPlaneGravity,
  runSimulation,
  simulateFlight,
  simulatePendulum,
  solveImpact,
  toFlightFrame,
  type SimulationInput,
} from "./simulation";

const MANUAL_INPUT: SimulationInput = {
  sourceKind: "manual",
  clubheadSpeedMph: 113.0,
  omegaDps: [0, 0, 0],
  loftDeg: 10.5,
  impactOffsetToeMm: 0,
  impactOffsetHighMm: 0,
  planeYawDeg: 0,
  planeSideTiltDeg: -45,
  planeForwardTiltDeg: 0,
  impactTimeS: null,
  swingDurationS: 1.5,
};

describe("pendulum parity (Python reference.py pins)", () => {
  it("matches the golf-default derived parameters", () => {
    const p = golfDefaultParams();
    expect(p.lc1).toBeCloseTo(0.3375, 12);
    expect(p.i1).toBeCloseTo(1.2058593750000002, 12);
    expect(p.lc2).toBeCloseTo(0.7557142857142858, 12);
    expect(p.i2).toBeCloseTo(0.24023500000000003, 12);
  });

  it("projects gravity into a -45 deg side-tilted plane like Python", () => {
    const [gx, gy] = inPlaneGravity(0, (-45 * Math.PI) / 180, 0);
    expect(gx).toBeCloseTo(0.0, 12);
    expect(gy).toBeCloseTo(-6.934348715723057, 9);
  });

  it("pins the RK4 state at steps 100 and 500 (dt = 1 ms)", () => {
    const g: [number, number] = [0.0, -6.934348715723057];
    const states = simulatePendulum(
      golfDefaultParams(),
      [-Math.PI / 2, 0, 0, 0],
      g,
      1e-3,
      500,
    );
    const s100 = states[100];
    expect(s100[0]).toBeCloseTo(-1.500595794495213, 9);
    expect(s100[1]).toBeCloseTo(-0.08686334675252501, 9);
    expect(s100[2]).toBeCloseTo(1.3853526740573519, 9);
    expect(s100[3]).toBeCloseTo(-1.6717820546064868, 9);
    const s500 = states[500];
    expect(s500[0]).toBeCloseTo(-0.14330846248913331, 8);
    expect(s500[1]).toBeCloseTo(-1.0243002763264488, 8);
    expect(s500[2]).toBeCloseTo(4.3312528971086515, 8);
    expect(s500[3]).toBeCloseTo(-0.06751810436510605, 8);
  });
});

describe("impact + launch parity (Python impact/models.py pins)", () => {
  const delivery = {
    clubheadSpeedMps: 50.51552,
    clubPathDeg: 2.0,
    faceAngleDeg: 0.0,
    attackAngleDeg: -1.5,
    dynamicLoftDeg: 10.5,
    impactOffsetToeMm: 0,
    impactOffsetHighMm: 0,
  };
  const impact = solveImpact(delivery);

  it("keeps the explicit default club identical to the legacy implicit default", () => {
    const explicit = solveImpact({ ...delivery, club: DEFAULT_IMPACT_CLUB });
    expect(explicit.ballVelocity).toEqual(impact.ballVelocity);
    expect(explicit.ballAngularVelocity).toEqual(impact.ballAngularVelocity);
  });

  it("independently uses selected head mass and MOI in the impact impulse", () => {
    const offCenter = { ...delivery, impactOffsetToeMm: 20 };
    const baseline = solveImpact({
      ...offCenter,
      club: { headMassKg: 0.2, moiAboutShaftKgM2: 4.5e-4 },
    });
    const heavier = solveImpact({
      ...offCenter,
      club: { headMassKg: 0.35, moiAboutShaftKgM2: 4.5e-4 },
    });
    const higherMoi = solveImpact({
      ...offCenter,
      club: { headMassKg: 0.2, moiAboutShaftKgM2: 1.2e-3 },
    });
    expect(Math.hypot(...heavier.ballVelocity)).toBeGreaterThan(
      Math.hypot(...baseline.ballVelocity),
    );
    expect(Math.hypot(...higherMoi.ballVelocity)).toBeGreaterThan(
      Math.hypot(...baseline.ballVelocity),
    );
  });

  it("pins the post-impact ball velocity", () => {
    expect(impact.ballVelocity[0]).toBeCloseTo(72.26017152461, 8);
    expect(impact.ballVelocity[1]).toBeCloseTo(13.392631176960073, 8);
    expect(impact.ballVelocity[2]).toBeCloseTo(0.0, 10);
  });

  it("pins the friction-spin vector (t x n axis, 2/7 cap)", () => {
    expect(impact.ballAngularVelocity[0]).toBeCloseTo(-10.752451814588115, 7);
    expect(impact.ballAngularVelocity[1]).toBeCloseTo(58.015038431649145, 7);
    expect(impact.ballAngularVelocity[2]).toBeCloseTo(351.43999531631596, 6);
  });

  it("pins the derived launch conditions", () => {
    const launch = deriveLaunch(
      toFlightFrame(impact.ballVelocity),
      toFlightFrame(impact.ballAngularVelocity),
    );
    expect(launch.ballSpeedMps).toBeCloseTo(73.49078145324175, 8);
    expect((launch.launchAngleRad * 180) / Math.PI).toBeCloseTo(10.5, 8);
    expect(launch.spinRpm).toBeCloseTo(3402.9736730363547, 6);
    expect(launch.spinAxis[0]).toBeCloseTo(-0.030173125408675523, 8);
    expect(launch.spinAxis[1]).toBeCloseTo(-0.9861976817154181, 8);
    expect(launch.spinAxis[2]).toBeCloseTo(0.16279961634539392, 8);
  });

  it("bands the Waterloo/Penner flight vs scipy RK45 (Python pins)", () => {
    const launch = deriveLaunch(
      toFlightFrame(impact.ballVelocity),
      toFlightFrame(impact.ballAngularVelocity),
    );
    const flight = simulateFlight(launch);
    // Python (solve_ivp RK45): carry 239.468 m, max height 26.193 m,
    // flight time 6.054 s, landing angle 33.885 deg, lateral 14.624 m.
    expect(flight.carryM).toBeGreaterThan(239.468 * 0.99);
    expect(flight.carryM).toBeLessThan(239.468 * 1.01);
    expect(flight.maxHeightM).toBeGreaterThan(26.193 * 0.98);
    expect(flight.maxHeightM).toBeLessThan(26.193 * 1.02);
    expect(flight.flightTimeS).toBeGreaterThan(6.054 * 0.99);
    expect(flight.flightTimeS).toBeLessThan(6.054 * 1.01);
    expect(flight.landingAngleDeg).toBeGreaterThan(33.885 - 1.0);
    expect(flight.landingAngleDeg).toBeLessThan(33.885 + 1.0);
  });
});

describe("session orchestration", () => {
  it("propagates selected club properties into launch results", () => {
    const offCenter = { ...MANUAL_INPUT, impactOffsetToeMm: 20 };
    const light = runSimulation({
      ...offCenter,
      club: { headMassKg: 0.15, moiAboutShaftKgM2: 2.0e-4 },
    });
    const heavy = runSimulation({
      ...offCenter,
      club: { headMassKg: 0.35, moiAboutShaftKgM2: 1.2e-3 },
    });
    expect(heavy.launch.ballSpeedMph).toBeGreaterThan(light.launch.ballSpeedMph);
    expect(heavy.launch.carryM).toBeGreaterThan(light.launch.carryM);
  });

  it("exports ball-aligned double-pendulum joints ending at the clubhead", () => {
    const run = runSimulation({ ...MANUAL_INPUT, sourceKind: "double_pendulum" });
    expect(run.sourceKind).toBe("double_pendulum");
    expect(run.swing[0].joints).toHaveLength(3);
    for (const sample of [run.swing[0], run.swing[500], run.swing[run.swing.length - 1]]) {
      const tip = sample.joints[sample.joints.length - 1];
      expect(tip[0]).toBeCloseTo(sample.position[0], 10);
      expect(tip[1]).toBeCloseTo(sample.position[1], 10);
      expect(tip[2]).toBeCloseTo(sample.position[2], 10);
    }
  });

  it("exports a four-point triple-pendulum skeleton", () => {
    const run = runSimulation({ ...MANUAL_INPUT, sourceKind: "triple_pendulum" });
    expect(run.swing[0].joints).toHaveLength(4);
    const sample = run.swing[500];
    const tip = sample.joints[sample.joints.length - 1];
    expect(tip[0]).toBeCloseTo(sample.position[0], 10);
    expect(tip[1]).toBeCloseTo(sample.position[1], 10);
    expect(tip[2]).toBeCloseTo(sample.position[2], 10);
  });

  it("manual run produces plausible driver numbers (Python band)", () => {
    const run = runSimulation(MANUAL_INPUT);
    expect(run.launch.ballSpeedMph).toBeGreaterThan(130);
    expect(run.launch.ballSpeedMph).toBeLessThan(185);
    expect(run.launch.launchAngleDeg).toBeGreaterThan(5);
    expect(run.launch.launchAngleDeg).toBeLessThan(20);
    expect(run.launch.spinRpm).toBeGreaterThan(1000);
    expect(run.launch.spinRpm).toBeLessThan(5000);
    expect(run.launch.carryM).toBeGreaterThan(150);
    expect(run.launch.carryM).toBeLessThan(320);
  });

  it("scrubbing tau makes the clubhead meet the fixed ball", () => {
    for (const tau of [0.01, 0.03, 0.045]) {
      const run = runSimulation({ ...MANUAL_INPUT, impactTimeS: tau });
      expect(run.impactTimeS).toBeCloseTo(tau, 9);
      const index = run.swing.findIndex(
        (sample) => Math.abs(sample.t - tau) < 5e-4,
      );
      const position = run.swing[index].position;
      expect(position[0]).toBeCloseTo(BALL_POSITION[0], 6);
      expect(position[1]).toBeCloseTo(BALL_POSITION[1], 6);
      expect(position[2]).toBeCloseTo(BALL_POSITION[2], 6);
    }
  });

  it("double-pendulum run swings toward the target and launches the ball", () => {
    const run = runSimulation({
      ...MANUAL_INPUT,
      sourceKind: "double_pendulum",
    });
    expect(run.launch.ballSpeedMph).toBeGreaterThan(0);
    expect(run.launch.carryM).toBeGreaterThan(0);
    expect(run.flight.length).toBeGreaterThan(2);
    expect(run.totalDurationS).toBeGreaterThan(run.impactTimeS);
  });

  it("flight starts at the ball position (app frame)", () => {
    const run = runSimulation(MANUAL_INPUT);
    expect(run.flight[0].position[0]).toBeCloseTo(BALL_POSITION[0], 9);
    expect(run.flight[0].position[1]).toBeCloseTo(BALL_POSITION[1], 9);
    expect(run.flight[0].position[2]).toBeCloseTo(BALL_POSITION[2], 9);
  });
});
