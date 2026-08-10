import { describe, expect, it } from "vitest";

import type { HeadMesh } from "../model/mesh";
import {
  computeMeshFaceShift,
  prepareShadedTriangles,
} from "./clubCanvasRendering";

const IDENTITY = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

describe("club canvas mesh rendering helpers", () => {
  it("aligns the furthest mesh face with the scenario face plane", () => {
    const mesh: HeadMesh = {
      triangles: [[[0.01, 0, 0], [0.06, 0, 0], [0.02, 0.01, 0]]],
      normals: [[0, 0, 1]],
    };

    const shift = computeMeshFaceShift(mesh, 40);
    expect(shift[0]).toBeCloseTo(-0.02);
    expect(shift.slice(1)).toEqual([0, 0]);
  });

  it("places, shades, and sorts mesh triangles from farthest to nearest", () => {
    const mesh: HeadMesh = {
      triangles: [
        [[0.3, 0, 0], [0.6, 0, 0], [0.3, 0.3, 0]],
        [[-0.6, 0, 0], [-0.3, 0, 0], [-0.6, 0.3, 0]],
      ],
      normals: [[0, 0, 1], [0, 0, 1]],
    };

    const shaded = prepareShadedTriangles({
      mesh,
      rotation: IDENTITY,
      shift: [0.1, 0, 0],
      offset: [0.2, 0, 0],
      yaw: 0,
      pitch: 0,
    });

    expect(shaded).toHaveLength(2);
    expect(shaded[0].depth).toBeCloseTo(-0.2);
    expect(shaded[1].depth).toBeCloseTo(0.7);
    expect(shaded[0].placed[0]).toEqual([-0.3, 0, 0]);
    expect(shaded[0].intensity).toBeGreaterThan(0.22);
  });
});
