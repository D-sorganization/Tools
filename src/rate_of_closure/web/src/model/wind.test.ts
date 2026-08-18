import { describe, expect, it } from "vitest";

import { simulateFlight, type Launch } from "./flight";
import fixture from "./__fixtures__/wind_scenario_golden_v1.json";
import { meteorologicalWind, windVelocityAt, type WindScenario } from "./wind";

const LAUNCH: Launch = {
  ballSpeedMps: 65,
  launchAngleRad: (12 * Math.PI) / 180,
  azimuthRad: 0,
  spinRpm: 2600,
  spinAxis: [0, -1, 0],
};

describe("wind scenario contract", () => {
  it("maps meteorological from-bearing into the flight frame", () => {
    expect(windVelocityAt(meteorologicalWind(10, 0), 0, [0, 0, 0]))
      .toEqual(expect.arrayContaining([expect.closeTo(-10, 12), 0, 0]));
    expect(windVelocityAt(meteorologicalWind(10, 90, 2), 0, [0, 0, 0]))
      .toEqual(expect.arrayContaining([expect.closeTo(0, 12), expect.closeTo(10, 12), 2]));
  });

  it("evaluates seeded turbulence as a pure deterministic function", () => {
    const scenario: WindScenario = {
      schemaVersion: "wind-scenario/v1",
      baseVelocityMps: [8, 1, 0],
      shearFractionPer10m: 0.1,
      gusts: [],
      turbulenceIntensityMps: 0.8,
      seed: 71,
      provenance: "test",
    };
    const first = windVelocityAt(scenario, 1.25, [10, 0, 20]);
    expect(windVelocityAt(scenario, 1.25, [10, 0, 20])).toEqual(first);
    expect(windVelocityAt(scenario, 1.35, [10, 0, 20])).not.toEqual(first);
  });

  it("changes the integrated path for headwind and tailwind", () => {
    const calm = simulateFlight(LAUNCH).carryM;
    const head = simulateFlight({ ...LAUNCH, windScenario: meteorologicalWind(10, 0) }).carryM;
    const tail = simulateFlight({ ...LAUNCH, windScenario: meteorologicalWind(10, 180) }).carryM;
    expect(head).toBeLessThan(calm);
    expect(tail).toBeGreaterThan(calm);
  });

  it("matches the shared Python/TypeScript golden wind field", () => {
    for (const golden of fixture.cases) {
      const scenario: WindScenario = {
        schemaVersion: "wind-scenario/v1",
        baseVelocityMps: golden.base_velocity_mps as [number, number, number],
        shearFractionPer10m: golden.shear_fraction_per_10m,
        gusts: golden.gusts.map((gust) => ({
          startTimeS: gust.start_time_s,
          durationS: gust.duration_s,
          peakVelocityMps: gust.peak_velocity_mps as [number, number, number],
        })),
        turbulenceIntensityMps: golden.turbulence_intensity_mps,
        seed: golden.seed,
        provenance: `golden:${golden.name}`,
      };
      const velocity = windVelocityAt(
        scenario,
        golden.time_s,
        golden.position_m as [number, number, number],
      );
      velocity.forEach((value, axis) => {
        // Kept in lockstep with the Python assertion in
        // `swing_sim/flight/tests/test_wind.py`: a parity contract has to state
        // one tolerance, not two. 12 decimals is below what the turbulence hash
        // reproduces across runtimes — it multiplies a sine by ~4.4e4 before
        // `% 1.0`, so a 1-ulp libm difference moves the phase ~4e-12. 9 decimals
        // is still eleven orders below any meaningful wind speed.
        expect(value).toBeCloseTo(golden.expected_velocity_mps[axis], 9);
      });
    }
  });
});
