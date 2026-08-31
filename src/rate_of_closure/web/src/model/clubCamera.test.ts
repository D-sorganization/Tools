import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/club_camera_golden_v1.json";
import {
  DEFAULT_CLUB_CAMERA,
  applyClubCameraAction,
  applyClubCameraDrag,
  cameraStatus,
  type ClubCamera,
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

  it("realizes every published field of the shared golden", () => {
    // Only `cases` was ever asserted, in either twin; `initial`, `limits`,
    // `orbit_step_deg` and `zoom_step` are published as the cross-runtime
    // contract yet went unenforced. Each is derived from the public API so
    // this pins observable behaviour, not a module-private literal.
    expect(fixture.schema).toBe("rate-of-closure/club-camera/v1");
    expect(DEFAULT_CLUB_CAMERA.azimuthDeg).toBeCloseTo(
      fixture.initial.azimuth_deg,
      12,
    );
    expect(DEFAULT_CLUB_CAMERA.elevationDeg).toBeCloseTo(
      fixture.initial.elevation_deg,
      12,
    );
    expect(DEFAULT_CLUB_CAMERA.zoom).toBeCloseTo(fixture.initial.zoom, 12);
    expect(fixture.cases.map((item) => item.action)).toEqual([
      "left",
      "right",
      "up",
      "down",
      "zoom_in",
      "zoom_out",
    ]);

    const stepped = applyClubCameraAction(DEFAULT_CLUB_CAMERA, "right");
    expect(stepped.azimuthDeg - DEFAULT_CLUB_CAMERA.azimuthDeg).toBeCloseTo(
      fixture.orbit_step_deg,
      12,
    );
    const zoomed = applyClubCameraAction(DEFAULT_CLUB_CAMERA, "zoom_in");
    expect(zoomed.zoom / DEFAULT_CLUB_CAMERA.zoom).toBeCloseTo(
      fixture.zoom_step,
      12,
    );

    const saturated: Record<string, ClubCamera> = {
      up: DEFAULT_CLUB_CAMERA,
      down: DEFAULT_CLUB_CAMERA,
      zoom_in: DEFAULT_CLUB_CAMERA,
      zoom_out: DEFAULT_CLUB_CAMERA,
    };
    for (let index = 0; index < 200; index += 1) {
      for (const action of Object.keys(saturated)) {
        saturated[action] = applyClubCameraAction(
          saturated[action],
          action as ClubCameraAction,
        );
      }
    }
    expect([saturated.down.elevationDeg, saturated.up.elevationDeg]).toEqual(
      fixture.limits.elevation_deg,
    );
    expect([saturated.zoom_out.zoom, saturated.zoom_in.zoom]).toEqual(
      fixture.limits.zoom,
    );
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
      expect(() => applyClubCameraDrag(DEFAULT_CLUB_CAMERA, value, 0)).toThrow(
        /finite/,
      );
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
