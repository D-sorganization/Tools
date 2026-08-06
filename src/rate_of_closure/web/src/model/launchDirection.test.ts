import { describe, expect, it } from "vitest";

import {
  convertLaunchDirection,
  launchDirectionFromRecord,
  launchDirectionToFlightAzimuth,
  migrateLaunchDirectionRecord,
  type LaunchDirectionConvention,
} from "./launchDirection";

const CONVENTIONS: LaunchDirectionConvention[] = [
  "app_native",
  "launch_monitor_comparable",
  "flight_frame",
];

describe("launch-direction conventions", () => {
  it.each([0, 7.25, -7.25, 90, -90, 179.999, -179.999])(
    "round-trips every convention pair at %s degrees",
    (degrees) => {
      for (const source of CONVENTIONS) {
        for (const target of CONVENTIONS) {
          const converted = convertLaunchDirection(degrees, source, target);
          expect(convertLaunchDirection(converted, target, source)).toBeCloseTo(degrees, 12);
        }
      }
    },
  );

  it("maps right-positive app direction to left-positive flight azimuth", () => {
    expect(launchDirectionToFlightAzimuth(6, "app_native")).toBe(-6);
  });
});

describe("launch-direction migration", () => {
  it("adds canonical fields without dropping legacy or unknown data", () => {
    const migrated = migrateLaunchDirectionRecord({
      launchAzimuthDeg: -3.5,
      shotName: "soft draw",
    });
    expect(migrated).toEqual({
      launchAzimuthDeg: -3.5,
      shotName: "soft draw",
      launchDirectionDeg: -3.5,
      launchDirectionConvention: "app_native",
      launchDirectionSchemaVersion: 1,
    });
    expect(launchDirectionFromRecord(migrated)).toEqual({
      degrees: -3.5,
      convention: "app_native",
    });
  });

  it("rejects conflicting canonical and legacy values", () => {
    expect(() =>
      migrateLaunchDirectionRecord({ launchDirectionDeg: 2, azimuthDeg: -2 }),
    ).toThrow(/conflicting launch-direction/);
  });

  it.each([Number.NaN, Number.POSITIVE_INFINITY, 181])("rejects invalid value %s", (value) => {
    expect(() => migrateLaunchDirectionRecord({ launchDirectionDeg: value })).toThrow();
  });
});
