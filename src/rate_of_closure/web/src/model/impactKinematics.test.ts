import { describe, expect, it } from "vitest";

import { getClub } from "./club";
import { DEFAULT_SCENARIO, solve } from "./impact";
import { exactEventSample, impactKinematics } from "./impactKinematics";
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
    expect(metrics.shaftAoaShapleyDeg).not.toBeNull();
    expect(metrics.shaftRotationRateDps).toBeCloseTo(1307, 10);
    const vectors = Object.fromEntries(metrics.vectors.map((vector) => [vector.key, vector.vectorMps]));
    const reconstructed = vectors.axisTranslation.map((value: number, index: number) =>
      value + vectors.shaftRotation[index] + vectors.otherRotation[index]);
    expect(reconstructed).toEqual(expect.arrayContaining(
      vectors.total.map((value: number) => expect.closeTo(value, 10)),
    ));
    expect(Math.abs(metrics.faceNormalUnit.reduce((sum, value, index) =>
      sum + value * metrics.leadingEdgeUnit[index], 0))).toBeLessThan(1e-10);
  });

  it("interpolates position and twist at an off-grid event", () => {
    const run = runSimulation(input);
    const eventIndex = 4;
    const eventTime = run.swing[eventIndex].t * 0.25 + run.swing[eventIndex + 1].t * 0.75;
    const exactRun = {
      ...run,
      impactTimeS: eventTime,
      swing: run.swing.map((sample, index) => ({
        ...sample,
        position: [index, 0, 0] as [number, number, number],
        velocity: [2 * index, 0, 0] as [number, number, number],
      })),
    };

    const sample = exactEventSample(exactRun);

    expect(sample.t).toBeCloseTo(eventTime, 12);
    expect(sample.position[0]).toBeCloseTo(eventIndex + 0.75, 12);
    expect(sample.velocity[0]).toBeCloseTo(2 * (eventIndex + 0.75), 12);
  });
});
