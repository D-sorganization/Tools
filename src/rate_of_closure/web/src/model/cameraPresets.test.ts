import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/camera_presets_v1.json";
import {
  CAMERA_COMMAND_IDS,
  applyCameraView,
  autoFitCamera,
  cameraPreset,
  canvasAngles,
  defaultCameraState,
  validateCameraPreset,
  validateCameraState,
  type CameraViewId,
} from "./cameraPresets";

describe("canonical camera preset contract", () => {
  it("matches every shared ID and exact frame orientation", () => {
    expect(fixture.frame).toEqual({ x: "downrange", y: "up", z: "right" });
    expect(CAMERA_COMMAND_IDS).toEqual(fixture.command_ids);
    for (const testCase of fixture.presets) {
      const preset = cameraPreset(testCase.command_id, testCase.face_on_side);
      expect(preset.viewDirection).toEqual(testCase.view_direction);
      expect(preset.screenUp).toEqual(testCase.screen_up);
      const angles = canvasAngles(preset);
      expect(angles.yawRad).toBeCloseTo(testCase.canvas_yaw_rad, 12);
      expect(angles.pitchRad).toBeCloseTo(testCase.canvas_pitch_rad, 12);
    }
  });

  it("fails closed for unknown IDs, bad sides, and invalid bounds", () => {
    expect(() => cameraPreset("camera.view.unknown" as CameraViewId, "right"))
      .toThrow(/unknown camera view/i);
    expect(() => cameraPreset("camera.view.face_on", "automatic" as "right"))
      .toThrow(/unknown face-on side/i);
    expect(() => autoFitCamera(defaultCameraState(), 0, 0.42)).toThrow(/positive/i);
    expect(() => validateCameraPreset({
      commandId: "camera.view.face_on",
      viewDirection: [1 + 5e-10, 0, 0],
      screenUp: [0, 1, 0],
    })).toThrow(/unit vector/i);
    expect(() => validateCameraState({
      ...defaultCameraState(), targetM: [Number.NaN, 0, 0],
    })).toThrow(/finite/i);
  });

  it("keeps preset changes idempotent and reserves scale changes for Auto Fit", () => {
    const initial = { ...defaultCameraState(), targetM: [1, 2, 3] as [number, number, number], zoom: 3.5 };
    const snapped = applyCameraView(initial, "camera.view.down_the_line");
    expect(applyCameraView(snapped, "camera.view.down_the_line")).toEqual(snapped);
    expect(snapped.targetM).toEqual(initial.targetM);
    expect(snapped.zoom).toBe(initial.zoom);
    const reset = applyCameraView(snapped, "camera.view.isometric");
    expect(reset.targetM).toEqual(initial.targetM);
    expect(reset.zoom).toBe(initial.zoom);
    const fitted = autoFitCamera(reset, 0.35, 0.42);
    expect(fitted.targetM).toEqual(initial.targetM);
    expect(fitted.presetId).toBe(reset.presetId);
    expect(fitted.zoom).not.toBe(reset.zoom);
    expect(0.35 * fitted.zoom).toBeLessThanOrEqual(0.42 * 0.84 + 1e-12);
  });
});
