/**
 * Parity pins for the standalone flight explorer (epic #4120, V2).
 *
 * The pinned tour-driver case mirrors
 * `tests/rate_of_closure/test_flight_explorer.py::TestLaunchFromDirect
 * ::test_pinned_tour_driver_case`; bands cover fixed-step RK4 (here)
 * vs scipy RK45 (Python), like the existing simulation parity tests.
 */

import { describe, expect, it } from "vitest";

import { directLaunch, exploreFlight } from "./flightExplorer";
import { BALL_POSITION } from "./simulation";

const PINNED = {
  ballSpeedMph: 167.0,
  launchAngleDeg: 10.9,
  azimuthDeg: 0.0,
  spinRpm: 2686.0,
  spinAxisTiltDeg: 0.0,
};

describe("directLaunch", () => {
  it("converts app-sign entries into flight-frame launch conditions", () => {
    const launch = directLaunch(PINNED);
    expect(launch.ballSpeedMps).toBeCloseTo(74.65568, 4);
    expect(launch.launchAngleRad).toBeCloseTo((10.9 * Math.PI) / 180, 12);
    expect(launch.azimuthRad).toBeCloseTo(0.0, 12);
    // Pure backspin: -y axis in the flight frame.
    expect(launch.spinAxis[0]).toBeCloseTo(0.0, 12);
    expect(launch.spinAxis[1]).toBeCloseTo(-1.0, 12);
    expect(launch.spinAxis[2]).toBeCloseTo(0.0, 12);
  });

  it("rejects non-positive ball speed", () => {
    expect(() => directLaunch({ ...PINNED, ballSpeedMph: 0 })).toThrow();
  });
});

describe("exploreFlight", () => {
  it("bands the pinned tour-driver case vs the Python pins", () => {
    // Python (scipy RK45, waterloo_penner): carry 247.484 m, apex
    // 28.226 m, time 6.278 s, landing 35.120 deg, lateral 0.0 m.
    const { metrics } = exploreFlight(directLaunch(PINNED));
    expect(metrics.carryM).toBeGreaterThan(247.484 * 0.99);
    expect(metrics.carryM).toBeLessThan(247.484 * 1.01);
    expect(metrics.maxHeightM).toBeGreaterThan(28.226 * 0.98);
    expect(metrics.maxHeightM).toBeLessThan(28.226 * 1.02);
    expect(metrics.flightTimeS).toBeGreaterThan(6.278 * 0.99);
    expect(metrics.flightTimeS).toBeLessThan(6.278 * 1.01);
    expect(metrics.landingAngleDeg).toBeGreaterThan(35.12 - 1.0);
    expect(metrics.landingAngleDeg).toBeLessThan(35.12 + 1.0);
    expect(Math.abs(metrics.lateralM)).toBeLessThan(0.5);
  });

  it("keeps app sign conventions: + azimuth and + tilt land right", () => {
    const right = exploreFlight(
      directLaunch({ ...PINNED, azimuthDeg: 5.0 }),
    ).metrics;
    const fade = exploreFlight(
      directLaunch({ ...PINNED, spinAxisTiltDeg: 10.0 }),
    ).metrics;
    const draw = exploreFlight(
      directLaunch({ ...PINNED, spinAxisTiltDeg: -10.0 }),
    ).metrics;
    expect(right.lateralM).toBeGreaterThan(1.0);
    expect(right.launchAzimuthDeg).toBeCloseTo(5.0, 8);
    expect(fade.lateralM).toBeGreaterThan(1.0);
    expect(draw.lateralM).toBeLessThan(-1.0);
  });

  it("returns an app-frame trajectory from the tee", () => {
    const { points } = exploreFlight(directLaunch(PINNED));
    expect(points.length).toBeGreaterThan(10);
    expect(points[0].position[0]).toBeCloseTo(BALL_POSITION[0], 6);
    expect(points[0].position[1]).toBeCloseTo(BALL_POSITION[1], 6);
    const last = points[points.length - 1];
    expect(last.position[0]).toBeGreaterThan(100.0);
    expect(last.position[1]).toBeLessThan(1.0); // back at the ground
  });
});
