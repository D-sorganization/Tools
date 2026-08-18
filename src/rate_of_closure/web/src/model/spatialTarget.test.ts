/** Cross-language spatial-target contract and golden tests (#4192). */

import { describe, expect, it } from "vitest";

import goldenFixture from "./__fixtures__/spatial_target_v1.json";
import {
  boxTolerance,
  createSpatialTarget,
  spatialTargetFromJson,
  spatialTargetMiss,
  spatialTargetMissFromFrame,
  spatialTargetFromRegion,
  spatialTargetToRegion,
  spatialTargetToJson,
  sphereTolerance,
  surfaceCircleTolerance,
  surfaceCorridorTolerance,
  targetPointFromFrame,
  targetPointInFrame,
} from "./targets";

function goldenTarget() {
  return createSpatialTarget({
    label: "Apex Gate",
    kind: "aerial_waypoint",
    point: targetPointFromFrame([137.5, 3.25, 24.25], "flight"),
    tolerance: boxTolerance([4.5, 2.5, 3.5]),
    elevationSource: "absolute",
  });
}

describe("spatial target frame and miss parity", () => {
  it("round trips through the flight frame", () => {
    const point = targetPointFromFrame([137.5, 3.25, 24.25], "flight");
    expect(point.appCoordinatesM).toEqual([137.5, 24.25, -3.25]);
    expect(targetPointInFrame(point, "flight")).toEqual([137.5, 3.25, 24.25]);
    expect(point.sourceFrame).toBe("flight");
  });

  it("returns the signed closest-point miss vector on all axes", () => {
    const target = createSpatialTarget({
      label: "Box",
      kind: "aerial_waypoint",
      point: targetPointFromFrame([100, 20, -5], "app"),
      tolerance: boxTolerance([2, 3, 4]),
      elevationSource: "absolute",
    });
    const miss = spatialTargetMiss(target, [105, 25, 2]);
    expect(miss.closestPointM).toEqual([102, 23, -1]);
    expect(miss.vectorM).toEqual([3, 2, 3]);
    expect(miss.distanceM).toBeCloseTo(Math.sqrt(22));
    expect(miss.accepted).toBe(false);

    const flightMiss = spatialTargetMissFromFrame(target, [105, -2, 25], "flight");
    expect(flightMiss).toEqual(miss);
  });

  it("uses radial closest points and accepts the sphere boundary", () => {
    const target = createSpatialTarget({
      label: "Sphere",
      kind: "aerial_waypoint",
      point: targetPointFromFrame([10, 20, 30], "app"),
      tolerance: sphereTolerance(2),
      elevationSource: "absolute",
    });
    expect(spatialTargetMiss(target, [11.2, 21.6, 30]).accepted).toBe(true);
    const outside = spatialTargetMiss(target, [13, 24, 30]);
    expect(outside.closestPointM).toEqual([11.2, 21.6, 30]);
    expect(outside.vectorM[0]).toBeCloseTo(1.8);
    expect(outside.vectorM[1]).toBeCloseTo(2.4);
    expect(outside.distanceM).toBeCloseTo(3);
  });

  it("keeps landing elevation explicit for surface geometry", () => {
    const target = createSpatialTarget({
      label: "Raised Green",
      kind: "landing_area",
      point: targetPointFromFrame([100, 5, 10], "app"),
      tolerance: surfaceCircleTolerance(10),
      elevationSource: "course_surface",
      groundSource: "course.surface/raised-green",
    });
    expect(spatialTargetMiss(target, [106, 5, 18]).accepted).toBe(true);
    expect(spatialTargetMiss(target, [106, 8, 18]).vectorM).toEqual([0, 3, 0]);
  });

  it("round trips the unchanged green and fairway 2D adapters", () => {
    for (const region of [
      {
        kind: "green" as const,
        distanceM: 180,
        radiusM: 12,
        lateralM: -4,
        bandHalfLengthM: 15,
        halfWidthM: 16,
      },
      {
        kind: "fairway" as const,
        distanceM: 220,
        radiusM: 10,
        lateralM: 0,
        bandHalfLengthM: 25,
        halfWidthM: 17,
      },
    ]) {
      const spatial = spatialTargetFromRegion(
        region,
        3.5,
        "course.surface/default",
      );
      expect(spatialTargetToRegion(spatial)).toEqual(region);
      expect(spatial.point.appCoordinatesM[1]).toBe(3.5);
    }
  });

  it("rejects incompatible kind geometry and non-finite coordinates", () => {
    expect(() =>
      createSpatialTarget({
        label: "Bad",
        kind: "landing_area",
        point: targetPointFromFrame([1, 0, 0], "app"),
        tolerance: sphereTolerance(1),
        elevationSource: "course_surface",
        groundSource: "surface",
      }),
    ).toThrow("surface tolerance");
    expect(() => targetPointFromFrame([1, Number.NaN, 3], "app")).toThrow("finite");
    expect(() => surfaceCorridorTolerance(10, 0)).toThrow("finite and > 0");
  });
});

describe("spatial target versioned serialization", () => {
  it("matches the shared Python/TypeScript golden byte-for-byte", () => {
    const golden = JSON.stringify(goldenFixture);
    const encoded = spatialTargetToJson(goldenTarget());
    expect(encoded).toBe(golden);
    expect(spatialTargetFromJson(encoded)).toEqual(goldenTarget());
    expect(spatialTargetToJson(spatialTargetFromJson(encoded))).toBe(encoded);
  });

  it("migrates legacy camelCase and snake_case 2D targets", () => {
    for (const legacy of [
      { kind: "green", distanceM: 230, radiusM: 10, lateralM: 4 },
      { kind: "green", distance_m: 230, radius_m: 10, lateral_m: 4 },
    ]) {
      const migrated = spatialTargetFromJson(JSON.stringify(legacy));
      expect(migrated.kind).toBe("landing_area");
      expect(migrated.point.appCoordinatesM).toEqual([230, 0, 4]);
      expect(migrated.groundSource).toBe("legacy.course_surface/default");
    }
  });

  it("rejects coercive legacy values", () => {
    expect(() =>
      spatialTargetFromJson('{"kind":"green","distance_m":true}'),
    ).toThrow("number");
  });

  it.each([
    ["units", (data: Record<string, unknown>) => { data.units = "ft"; }],
    ["frame", (data: Record<string, unknown>) => { data.frame = "flight"; }],
    ["schema_version", (data: Record<string, unknown>) => { data.schema_version = 2; }],
    ["unknown fields", (data: Record<string, unknown>) => { data.extra = true; }],
  ])("rejects invalid %s", (message, mutate) => {
    const data = JSON.parse(spatialTargetToJson(goldenTarget())) as Record<string, unknown>;
    mutate(data);
    expect(() => spatialTargetFromJson(JSON.stringify(data))).toThrow(message);
  });
});
