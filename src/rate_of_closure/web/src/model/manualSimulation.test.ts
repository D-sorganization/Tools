import { describe, expect, it } from "vitest";

import { add } from "./impactPhysics";
import { DEFAULT_SCENARIO } from "./impact";
import { exactEventSample, impactKinematics } from "./impactKinematics";
import { getClub } from "./club";
import { faceCenterPoint, hoselPoint } from "./clubHeads";
import { applyRotation } from "./rotation";
import { runSimulation, type SimulationInput, type SimulationRunTs } from "./simulation";

const INPUT: SimulationInput = {
  sourceKind: "manual",
  clubheadSpeedMph: 30,
  omegaDps: [0, 100, 0],
  loftDeg: 46,
  impactOffsetToeMm: 0,
  impactOffsetHighMm: 0,
  planeYawDeg: 0,
  planeSideTiltDeg: -45,
  planeForwardTiltDeg: 0,
  impactTimeS: 0.03,
  swingDurationS: 1.5,
};

const requireLaunch = (run: SimulationRunTs) => {
  if (!run.launch) throw new Error("expected launch");
  return run.launch;
};

describe("manual three-dimensional delivery", () => {
  it("propagates signed attack angle and club path into reference velocity", () => {
    const run = runSimulation({
      ...INPUT,
      manualAttackAngleDeg: -10,
      manualClubPathDeg: 6,
    });
    const velocity = run.swing[30].velocity;
    expect(Math.hypot(...velocity) * 2.2369362920544).toBeCloseTo(30, 10);
    expect(Math.atan2(velocity[1], Math.hypot(velocity[0], velocity[2])) * 180 / Math.PI)
      .toBeCloseTo(-10, 10);
    expect(Math.atan2(velocity[2], velocity[0]) * 180 / Math.PI).toBeCloseTo(6, 10);
  });

  it("applies targetward-positive forward lean to pose, angular velocity, and dynamic loft", () => {
    const leanRad = 15 * Math.PI / 180;
    const run = runSimulation({ ...INPUT, manualForwardShaftLeanDeg: 15 });
    const sample = run.swing[30];
    expect(sample.rotation[0][0]).toBeCloseTo(Math.cos(leanRad), 12);
    expect(sample.rotation[0][1]).toBeCloseTo(Math.sin(leanRad), 12);
    expect(sample.rotation[1][0]).toBeCloseTo(-Math.sin(leanRad), 12);
    expect(sample.angularVelocity[0]).toBeCloseTo(Math.sin(leanRad) * 100 * Math.PI / 180, 12);
    expect(sample.angularVelocity[1]).toBeCloseTo(Math.cos(leanRad) * 100 * Math.PI / 180, 12);
    expect(requireLaunch(run).launchAngleDeg).toBeCloseTo(31, 10);
  });

  it("uses the selected generated-head hosel as the manual shaft-axis datum", () => {
    const club = getClub("Pitching Wedge");
    const run = runSimulation({
      ...INPUT,
      club,
      manualForwardShaftLeanDeg: 15,
      shaftAxisDatum: "generated_hosel",
    });
    const sample = exactEventSample(run);
    const metrics = impactKinematics(run, DEFAULT_SCENARIO, club);
    const registeredHoselLever = add(
      [DEFAULT_SCENARIO.comToFaceMm / 1000, 0, 0],
      hoselPoint(club).map((value, index) =>
        value - faceCenterPoint(club)[index]) as [number, number, number],
    );
    const expected = add(
      sample.position,
      applyRotation(sample.rotation, registeredHoselLever),
    );

    expect(metrics.geometryBasis).toBe("generated_head_profile_hosel");
    metrics.shaftAxisPointM.forEach((value, index) =>
      expect(value).toBeCloseTo(expected[index], 12));
    expect(metrics.shaftAxisPointM).not.toEqual(metrics.referencePointM);
  });
});
