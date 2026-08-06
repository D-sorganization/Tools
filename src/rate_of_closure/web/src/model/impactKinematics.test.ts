import { describe, expect, it } from "vitest";

import { getClub } from "./club";
import { DEFAULT_SCENARIO, solve } from "./impact";
import { impactKinematics } from "./impactKinematics";
import { runSimulation, type SimulationInput } from "./simulation";

const scenario = {
  ...DEFAULT_SCENARIO,
  clubheadSpeedMph: 30,
  lieAngleDeg: 64,
  omegaPlaneDps: 0,
  omegaShaftDps: 1307,
  comToFaceMm: 20,
};

const input: SimulationInput = {
  sourceKind: "manual",
  clubheadSpeedMph: scenario.clubheadSpeedMph,
  omegaDps: solve(scenario).omegaDps,
  loftDeg: 46,
  impactOffsetToeMm: 0,
  impactOffsetHighMm: 0,
  planeYawDeg: 0,
  planeSideTiltDeg: -45,
  planeForwardTiltDeg: 0,
  impactTimeS: 0.03,
  swingDurationS: 1.5,
};

describe("impact kinematics", () => {
  it("reconciles the manual rigid-body point-velocity fixture", () => {
    const metrics = impactKinematics(
      runSimulation(input), scenario, getClub("Pitching Wedge"),
    );
    const expected = solve(scenario);

    expect(metrics.eventLabel).toBe("Impact");
    expect(metrics.contactAoaDeg).toBeCloseTo(expected.aoaDeviationDeg, 10);
    expect(metrics.shaftAoaContributionDeg).toBeLessThan(0);
    expect(metrics.shaftRotationRateDps).toBeCloseTo(1307, 10);
  });
});
