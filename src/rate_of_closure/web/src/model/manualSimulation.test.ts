import { describe, expect, it } from "vitest";

import { add } from "./impactPhysics";
import { DEFAULT_SCENARIO, solve } from "./impact";
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
  it("rejects invalid delivered dynamic loft before running the simulation", () => {
    expect(() => runSimulation({
      ...INPUT,
      loftDeg: 46,
      manualForwardShaftLeanDeg: -44,
    })).toThrow(/delivered dynamic loft.*90.*\[-89, 89\]/i);
  });
  it.each(["double_pendulum", "triple_pendulum"] as const)(
    "does not apply inactive manual-loft validation to %s",
    (sourceKind) => {
      expect(() => runSimulation({
        ...INPUT,
        sourceKind,
        loftDeg: 46,
        manualForwardShaftLeanDeg: -44,
        swingDurationS: 0.05,
      })).not.toThrow();
    },
  );
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
    expect(metrics.modelLimitations).toMatch(/representative generated head-profile hosel/i);
    expect(metrics.modelLimitations).toMatch(/shaft attribution is a kinematic analysis/i);
    expect(metrics.modelLimitations).toMatch(/forced contact aligns.*reference point/i);
    metrics.shaftAxisPointM.forEach((value, index) =>
      expect(value).toBeCloseTo(expected[index], 12));
    expect(metrics.shaftAxisPointM).not.toEqual(metrics.referencePointM);
  });

  it("states the same kinematic-only boundary for the legacy manual datum", () => {
    const metrics = impactKinematics(
      runSimulation(INPUT),
      DEFAULT_SCENARIO,
      getClub("Pitching Wedge"),
    );
    expect(metrics.geometryBasis).toBe("scenario_shaft_line");
    expect(metrics.modelLimitations).toMatch(/tracked-reference translation/i);
    expect(metrics.modelLimitations).toMatch(/not shaft-induced contact-point velocity/i);
    expect(metrics.modelLimitations).toMatch(/forced contact aligns.*reference point/i);
  });

  it("pins the representative 30 mph pitching-wedge decomposition", () => {
    const club = getClub("Pitching Wedge");
    const scenario = {
      ...DEFAULT_SCENARIO,
      clubheadSpeedMph: 30,
      omegaPlaneDps: 0,
      omegaShaftDps: 1307,
      lieAngleDeg: 64,
      comToFaceMm: 20,
    };
    const run = runSimulation({
      ...INPUT,
      club,
      clubheadSpeedMph: scenario.clubheadSpeedMph,
      omegaDps: solve(scenario).omegaDps,
      loftDeg: club.loftDeg,
      manualAttackAngleDeg: -10,
      manualClubPathDeg: 0,
      manualForwardShaftLeanDeg: 15,
      shaftAxisDatum: "generated_hosel",
    });

    const metrics = impactKinematics(run, scenario, club);
    const contactVelocity = metrics.vectors.find(({ key }) => key === "total")
      ?.vectorMps;
    if (!contactVelocity) throw new Error("expected total contact velocity vector");

    expect(contactVelocity[0]).toBeCloseTo(13.155691, 6);
    expect(contactVelocity[1]).toBeCloseTo(-2.522013, 6);
    expect(contactVelocity[2]).toBeCloseTo(-0.410056, 6);
    expect(metrics.contactAoaDeg).toBeCloseTo(-10.847087, 6);
    // Shaft-counterfactual decomposition repinned for #4799 G2: the
    // loft-aware wedge hosel moved the shaft axis to the leading edge,
    // changing the lever arm. Total delivery and carry are unchanged.
    expect(metrics.withoutShaftAoaDeg).toBeCloseTo(-11.270053, 6);
    expect(metrics.shaftAoaContributionDeg).toBeCloseTo(0.422965, 6);
    expect(metrics.shaftVerticalVelocityShare).toBeCloseTo(0.018553, 6);
    expect(requireLaunch(run).carryM).toBeCloseTo(22.45855, 4);

    const contactTargetRun = runSimulation({
      ...INPUT,
      club,
      clubheadSpeedMph: scenario.clubheadSpeedMph,
      omegaDps: solve(scenario).omegaDps,
      loftDeg: club.loftDeg,
      manualAttackAngleDeg: -9.1535118584,
      manualForwardShaftLeanDeg: 15,
      shaftAxisDatum: "generated_hosel",
    });
    const contactTargetMetrics = impactKinematics(contactTargetRun, scenario, club);
    expect(contactTargetMetrics.contactAoaDeg).toBeCloseTo(-10, 8);
    // Repinned for #4799 G2 (see above): lever arm from the new hosel.
    expect(contactTargetMetrics.shaftAoaContributionDeg).toBeCloseTo(0.373815, 6);
    expect(requireLaunch(contactTargetRun).carryM).toBeCloseTo(23.024061, 4);
  });
});
