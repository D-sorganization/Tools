import { describe, expect, it } from "vitest";

import { analyzeDPlane, spinLoftSectorDirections } from "./dPlane";
import { dot, norm, type Vec3 } from "./impactPhysics";
import goldenFixture from "./__fixtures__/dplane_golden_v1.json";

const direction = (headingDeg: number, elevationDeg: number): Vec3 => {
  const heading = headingDeg * Math.PI / 180;
  const elevation = elevationDeg * Math.PI / 180;
  return [
    Math.cos(elevation) * Math.cos(heading),
    Math.sin(elevation),
    Math.cos(elevation) * Math.sin(heading),
  ];
};

describe("3-D D-plane analytic cases", () => {
  it("matches the Python/TypeScript golden contract", () => {
    const propertyMap = {
      spin_loft_3d_deg: "spinLoft3dDeg",
      planar_spin_loft_deg: "planarSpinLoftDeg",
      signed_planar_gap_deg: "signedPlanarGapDeg",
      spin_loft_residual_deg: "spinLoftResidualDeg",
      club_path_deg: "clubPathDeg",
      attack_angle_deg: "attackAngleDeg",
      face_angle_deg: "faceAngleDeg",
      dynamic_loft_deg: "dynamicLoftDeg",
      face_to_path_deg: "faceToPathDeg",
      dplane_normal_azimuth_deg: "dplaneNormalAzimuthDeg",
      dplane_tilt_deg: "dplaneTiltDeg",
      dplane_inclination_deg: "dplaneInclinationDeg",
      ground_intersection_azimuth_deg: "groundIntersectionAzimuthDeg",
    } as const;

    expect(goldenFixture.schema_version).toBe("dplane-golden-v1");
    for (const fixtureCase of goldenFixture.cases) {
      const result = analyzeDPlane(
        fixtureCase.travel_vector as Vec3,
        fixtureCase.face_normal as Vec3,
        undefined,
        undefined,
        goldenFixture.frame_id,
      );
      expect(result.status, fixtureCase.id).toBe(fixtureCase.expected.status);
      for (const [fixtureName, resultName] of Object.entries(propertyMap)) {
        expect(
          result[resultName],
          `${fixtureCase.id}: ${fixtureName}`,
        ).toBeCloseTo(
          fixtureCase.expected[fixtureName as keyof typeof fixtureCase.expected] as number,
          9,
        );
      }
    }
  });

  it("matches the square descending planar fixture", () => {
    const result = analyzeDPlane(direction(0, -3), direction(0, 12));
    expect(result.status).toBe("defined");
    expect(result.spinLoft3dDeg).toBeCloseTo(15, 10);
    expect(result.planarSpinLoftDeg).toBeCloseTo(15, 10);
    expect(result.spinLoftResidualDeg).toBeCloseTo(0, 10);
    expect(result.dplaneNormalUnit).toEqual(expect.arrayContaining([
      expect.closeTo(0, 10), expect.closeTo(0, 10), expect.closeTo(1, 10),
    ]));
  });

  it("reports the full-3D residual when horizontal headings differ", () => {
    const result = analyzeDPlane(direction(-2, -5), direction(6, 31));
    expect(result.faceToPathDeg).toBeCloseTo(8, 10);
    expect(result.planarSpinLoftDeg).toBeCloseTo(36, 10);
    expect(result.spinLoft3dDeg!).toBeGreaterThan(36);
    expect(result.spinLoftResidualDeg).toBeCloseTo(result.spinLoft3dDeg! - 36, 10);
  });

  it("constructs a unit sector entirely inside the D-plane", () => {
    const result = analyzeDPlane(direction(3, -4), direction(-2, 28));
    const sector = spinLoftSectorDirections(result, 12);
    expect(sector).toHaveLength(13);
    expect(sector[0]).toEqual(expect.arrayContaining(
      result.travelDirectionUnit!.map((value) => expect.closeTo(value, 10)),
    ));
    expect(sector[12]).toEqual(expect.arrayContaining(
      result.faceNormalUnit.map((value) => expect.closeTo(value, 10)),
    ));
    for (const vector of sector) {
      expect(norm(vector)).toBeCloseTo(1, 10);
      expect(dot(result.dplaneNormalUnit!, vector)).toBeCloseTo(0, 10);
    }
  });
});

describe("D-plane typed singular states", () => {
  it("does not invent a plane for zero travel", () => {
    const result = analyzeDPlane([0, 0, 0], direction(4, 20));
    expect(result.status).toBe("zero_travel");
    expect(result.travelDirectionUnit).toBeNull();
    expect(result.dplaneNormalUnit).toBeNull();
    expect(result.spinLoft3dDeg).toBeNull();
    expect(result.faceAngleDeg).toBeCloseTo(4, 10);
  });

  it.each([
    [[1, 0, 0] as Vec3, "parallel", 0],
    [[-1, 0, 0] as Vec3, "antiparallel", 180],
  ])("types collinear vectors as %s", (face, status, spinLoft) => {
    const result = analyzeDPlane([1, 0, 0], face);
    expect(result.status).toBe(status);
    expect(result.spinLoft3dDeg).toBeCloseTo(spinLoft, 10);
    expect(result.dplaneNormalUnit).toBeNull();
    expect(spinLoftSectorDirections(result)).toEqual([]);
  });

  it("marks vertical projected headings unavailable", () => {
    const result = analyzeDPlane([0, 1, 0], direction(5, 20));
    expect(result.clubPathDeg).toBeNull();
    expect(result.faceToPathDeg).toBeNull();
    expect(result.attackAngleDeg).toBeCloseTo(90, 10);
  });
});

describe("D-plane transformation behavior", () => {
  it("preserves intrinsic results under a common proper rotation", () => {
    const travel = direction(-4, -7);
    const face = direction(3, 25);
    const baseline = analyzeDPlane(travel, face);
    const angle = 37 * Math.PI / 180;
    const rotate = ([x, y, z]: Vec3): Vec3 => [
      Math.cos(angle) * x + Math.sin(angle) * z,
      y,
      -Math.sin(angle) * x + Math.cos(angle) * z,
    ];
    const rotated = analyzeDPlane(rotate(travel), rotate(face), rotate([1, 0, 0]), rotate([0, 1, 0]));
    expect(rotated.spinLoft3dDeg).toBeCloseTo(baseline.spinLoft3dDeg!, 10);
    expect(rotated.planarSpinLoftDeg).toBeCloseTo(baseline.planarSpinLoftDeg!, 10);
    expect(rotated.faceToPathDeg).toBeCloseTo(baseline.faceToPathDeg!, 10);
    expect(rotated.dplaneTiltDeg).toBeCloseTo(baseline.dplaneTiltDeg!, 10);
  });

  it("reverses signed horizontal values under right-left reflection", () => {
    const travel = direction(-3, -6);
    const face = direction(5, 24);
    const original = analyzeDPlane(travel, face);
    const reflect = ([x, y, z]: Vec3): Vec3 => [x, y, -z];
    const reflected = analyzeDPlane(reflect(travel), reflect(face));
    expect(reflected.clubPathDeg).toBeCloseTo(-original.clubPathDeg!, 10);
    expect(reflected.faceAngleDeg).toBeCloseTo(-original.faceAngleDeg!, 10);
    expect(reflected.faceToPathDeg).toBeCloseTo(-original.faceToPathDeg!, 10);
    expect(reflected.dplaneTiltDeg).toBeCloseTo(-original.dplaneTiltDeg!, 10);
    expect(reflected.spinLoft3dDeg).toBeCloseTo(original.spinLoft3dDeg!, 10);
  });
});
