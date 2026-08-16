import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/club_camera_golden_v1.json";
import {
  DEFAULT_CLUB_CAMERA,
  applyClubCameraAction,
  applyClubCameraDrag,
  cameraStatus,
  type ClubCameraAction,
} from "./clubCamera";

describe("club camera shared contract", () => {
  it("matches the Python-owned transition golden", () => {
    for (const item of fixture.cases) {
      const actual = applyClubCameraAction(
        DEFAULT_CLUB_CAMERA,
        item.action as ClubCameraAction,
      );
      expect(actual.azimuthDeg).toBeCloseTo(item.expected.azimuth_deg, 12);
      expect(actual.elevationDeg).toBeCloseTo(item.expected.elevation_deg, 12);
      expect(actual.zoom).toBeCloseTo(item.expected.zoom, 12);
    }
  });

  it("clamps and resets exactly", () => {
    let camera = DEFAULT_CLUB_CAMERA;
    for (let index = 0; index < 100; index += 1) {
      camera = applyClubCameraAction(camera, "up");
      camera = applyClubCameraAction(camera, "zoom_in");
    }
    expect(camera).toMatchObject({ elevationDeg: 80, zoom: 4 });
    expect(applyClubCameraAction(camera, "home")).toEqual(DEFAULT_CLUB_CAMERA);
  });

  it("formats one visible camera status", () => {
    expect(cameraStatus(DEFAULT_CLUB_CAMERA, "Procedural head")).toBe(
      "Procedural head; camera azimuth 150°, elevation 30°, zoom 1.00×.",
    );
  });

  it.each([Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY])(
    "rejects nonfinite public-boundary value %s",
    (value) => {
      const forged = { azimuthDeg: value, elevationDeg: 0, zoom: 1 };
      expect(() => applyClubCameraAction(forged, "left")).toThrow(/finite/);
      expect(() => cameraStatus(forged, "source")).toThrow(/finite/);
      expect(() => applyClubCameraDrag(DEFAULT_CLUB_CAMERA, value, 0)).toThrow(/finite/);
    },
  );

  it("keeps long action and drag sequences canonical", () => {
    let camera = DEFAULT_CLUB_CAMERA;
    for (let index = 0; index < 10_000; index += 1) {
      camera = applyClubCameraAction(camera, "right");
      camera = applyClubCameraDrag(camera, -1, 1);
    }
    expect(camera.azimuthDeg).toBeGreaterThanOrEqual(-180);
    expect(camera.azimuthDeg).toBeLessThan(180);
    expect(camera.elevationDeg).toBe(80);
  });
});
