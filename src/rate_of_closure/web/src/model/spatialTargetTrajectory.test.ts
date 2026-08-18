import { describe, expect, it } from "vitest";

import type { FlightPoint } from "./flight";
import {
  boxTolerance,
  createSpatialTarget,
  sphereTolerance,
  targetPointFromFrame,
} from "./spatialTarget";
import { assessSpatialTargetTrajectory } from "./spatialTargetTrajectory";

const point = (time: number, position: [number, number, number]): FlightPoint => ({
  time, position, velocity: [1, 0, 0],
});

const waypoint = (tolerance: ReturnType<typeof sphereTolerance> | ReturnType<typeof boxTolerance>) =>
  createSpatialTarget({
    label: "Window",
    kind: "aerial_waypoint",
    point: targetPointFromFrame([5, 0, 0], "app"),
    tolerance,
    elevationSource: "absolute",
  });

describe("continuous spatial-target trajectory assessment", () => {
  it("detects a sphere crossed strictly between solver samples", () => {
    const result = assessSpatialTargetTrajectory(
      waypoint(sphereTolerance(0.5)),
      [point(0, [0, 0, 0]), point(2, [10, 0, 0])],
    );

    expect(result.miss.accepted).toBe(true);
    expect(result.timeS).toBeCloseTo(1);
    expect(result.segmentIndex).toBe(0);
    expect(result.fraction).toBeCloseTo(0.5);
  });

  it("detects a box crossed strictly between solver samples", () => {
    const result = assessSpatialTargetTrajectory(
      waypoint(boxTolerance([0.5, 1, 1])),
      [point(3, [0, 0.75, 0.5]), point(5, [10, 0.75, 0.5])],
    );

    expect(result.miss.accepted).toBe(true);
    expect(result.timeS).toBeCloseTo(4);
    expect(result.actualPointM).toEqual([5, 0.75, 0.5]);
  });
});
