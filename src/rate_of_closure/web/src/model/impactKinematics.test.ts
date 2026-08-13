import { describe, expect, it } from "vitest";

import { getClub } from "./club";
import sashoFixture from "./__fixtures__/sasho_face_center_rotation_golden_v1.json";
import { add, cross, norm, sub } from "./impactPhysics";
import { DEFAULT_SCENARIO, solve } from "./impact";
import {
  SASHO_FACE_CENTER_ROTATION_METHOD_ID,
  exactEventSample,
  impactKinematics,
  sashoFaceCenterRotationAoa,
} from "./impactKinematics";
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
  it("matches the Python-owned Sasho nearest-shaft face-center golden", () => {
    const result = sashoFaceCenterRotationAoa(
      sashoFixture.angular_velocity_rad_s as [number, number, number],
      sashoFixture.shaft_axis_point_m as [number, number, number],
      sashoFixture.shaft_axis_unit as [number, number, number],
      sashoFixture.face_center_point_m as [number, number, number],
    );

    expect(result.methodId).toBe(SASHO_FACE_CENTER_ROTATION_METHOD_ID);
    expect(result.methodId).toBe(sashoFixture.expected.method_id);
    result.nearestShaftPointM.forEach((value, index) =>
      expect(value).toBeCloseTo(sashoFixture.expected.nearest_shaft_point_m[index], 12));
    result.leverArmM.forEach((value, index) =>
      expect(value).toBeCloseTo(sashoFixture.expected.lever_arm_m[index], 12));
    result.velocityMps.forEach((value, index) =>
      expect(value).toBeCloseTo(sashoFixture.expected.velocity_mps[index], 12));
    expect(result.aoaDeg).toBeCloseTo(sashoFixture.expected.aoa_deg, 12);
  });

  it("fails closed on invalid Sasho geometry", () => {
    expect(() => sashoFaceCenterRotationAoa(
      [Number.NaN, 10, -4], [0, 0, 0], [0, 1, 0], [0.02, 0.1, 0.03],
    )).toThrow(/angular velocity/i);
    expect(() => sashoFaceCenterRotationAoa(
      [2, 10, -4], [0, 0, 0], [0, 0, 0], [0.02, 0.1, 0.03],
    )).toThrow(/shaft axis/i);
  });

  it("reconciles the manual rigid-body point-velocity fixture", () => {
    const metrics = impactKinematics(
      runSimulation(input), scenario, getClub("Pitching Wedge"),
    );
    const expected = solve(scenario);

    expect(metrics.eventLabel).toBe("Impact");
    expect(metrics.contactAoaDeg).toBeCloseTo(expected.aoaDeviationDeg, 10);
    expect(metrics.shaftAoaContributionDeg).toBeLessThan(0);
    expect(metrics.shaftRotationRateDps).toBeCloseTo(1307, 10);
    expect(metrics.shaftShapleyAoaDeg).not.toBeNull();
    expect(metrics.sashoFaceCenterRotation.methodId).toBe(
      SASHO_FACE_CENTER_ROTATION_METHOD_ID,
    );
    expect(metrics.sashoFaceCenterRotation.velocityMps).not.toEqual(
      metrics.vectors.find((vector) => vector.key === "shaftRotation")!.vectorMps,
    );
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

  it("uses rigid-body face-center travel and location-dependent curved-face geometry", () => {
    const curvedScenario = {
      ...scenario,
      impactOffsetToeMm: 12,
      impactOffsetHighMm: 5,
    };
    const run = runSimulation({
      ...input,
      impactOffsetToeMm: 12,
      impactOffsetHighMm: 5,
    });
    const sample = exactEventSample(run);
    const metrics = impactKinematics(run, curvedScenario, getClub("Driver 10.5°"));
    const lever = sub(metrics.faceCenterPointM, sample.position);
    const expectedVelocity = add(sample.velocity, cross(sample.angularVelocity, lever));

    expectedVelocity.forEach((value, index) =>
      expect(metrics.faceCenterVelocityMps[index]).toBeCloseTo(value, 10));
    const expectedDirection = expectedVelocity.map((value) =>
      value / norm(expectedVelocity)) as [number, number, number];
    expectedDirection.forEach((value, index) =>
      expect(metrics.faceCenterDPlane.travelDirectionUnit![index]).toBeCloseTo(value, 10));
    expect(metrics.contactDPlane.faceAngleDeg).not.toBeCloseTo(
      metrics.faceCenterDPlane.faceAngleDeg!, 6,
    );
    expect(metrics.faceCenterDPlane.spinLoft3dDeg).not.toBeNull();
    expect(metrics.faceCenterDPlane.planarSpinLoftDeg).not.toBeNull();
  });
});
