import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/confidence_ellipsoid_mesh_golden_v1.json";
import {
  buildConfidenceEllipsoidMesh,
  MAX_ELLIPSOID_TRIANGLES,
  MAX_ELLIPSOID_VERTICES,
  MAX_RENDERED_ELLIPSOIDS,
  type ConfidenceEllipsoidGeometryTs,
} from "./confidenceEllipsoidMesh";

describe("confidence ellipsoid mesh", () => {
  it("matches the Python golden and excludes rank-deficient samples", () => {
    const mesh = buildConfidenceEllipsoidMesh(
      fixture as unknown as ConfidenceEllipsoidGeometryTs,
      fixture.budget,
    );
    expect(mesh.sampleIndices).toEqual(fixture.sampleIndices);
    mesh.verticesM.forEach((vertex, index) => {
      vertex.forEach((value, component) => {
        expect(value).toBeCloseTo(fixture.verticesM[index][component], 14);
      });
    });
    expect(mesh.triangles).toEqual(fixture.triangles);
  });

  it("bounds dense studies while retaining the temporal endpoints", () => {
    const count = 1_000;
    const mesh = buildConfidenceEllipsoidMesh({
      centersM: Array.from({ length: count }, (_, index) => [index, 0, 0]),
      principalFrames: Array.from({ length: count }, () =>
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
      semiAxisLengthsM: Array.from({ length: count }, () => [1, 1, 1]),
      adequacy: Array.from({ length: count }, () => "estimable"),
      coordinateFrame: "app_frame:x_target,y_up,z_right",
    });
    expect(mesh.sampleIndices).toHaveLength(MAX_RENDERED_ELLIPSOIDS);
    expect(mesh.sampleIndices[0]).toBe(0);
    expect(mesh.sampleIndices[mesh.sampleIndices.length - 1]).toBe(count - 1);
    expect(mesh.verticesM.length).toBeLessThanOrEqual(MAX_ELLIPSOID_VERTICES);
    expect(mesh.triangles.length).toBeLessThanOrEqual(MAX_ELLIPSOID_TRIANGLES);
  });

  it("fails closed for malformed estimable geometry", () => {
    const valid: ConfidenceEllipsoidGeometryTs = {
      centersM: [[0, 0, 0]],
      principalFrames: [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
      semiAxisLengthsM: [[1, 1, 1]],
      adequacy: ["estimable"],
      coordinateFrame: "app_frame:x_target,y_up,z_right",
    };
    expect(() => buildConfidenceEllipsoidMesh({ ...valid, coordinateFrame: "other" })).toThrow();
    expect(() => buildConfidenceEllipsoidMesh({
      ...valid, principalFrames: [[[2, 0, 0], [0, 1, 0], [0, 0, 1]]],
    })).toThrow(/orthonormal/);
    expect(() => buildConfidenceEllipsoidMesh({
      ...valid, semiAxisLengthsM: [[1, 0, 1]],
    })).toThrow(/positive/);
  });
});
