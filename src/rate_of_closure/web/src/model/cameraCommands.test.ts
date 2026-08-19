import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/camera_commands_v1.json";
import {
  CAMERA_COMMAND_IDS,
  applyManualOverride,
  cameraPreset,
  canvasAngles,
  recenterCamera,
  safeTrackingZoom,
  setTrackingEnabled,
  updateTrackingTarget,
  type CameraState,
  type CameraViewId,
  type FaceOnSide,
} from "./cameraCommands";

describe("camera command contract", () => {
  it("matches every shared command ID and exact canonical orientation", () => {
    expect(CAMERA_COMMAND_IDS).toEqual(fixture.command_ids);
    for (const testCase of fixture.presets) {
      const preset = cameraPreset(
        testCase.command_id as CameraViewId,
        testCase.face_on_side as FaceOnSide,
      );
      expect(preset.viewDirection).toEqual(testCase.view_direction);
      expect(preset.screenUp).toEqual(testCase.screen_up);
      const angles = canvasAngles(preset);
      expect(angles.yawRad).toBeCloseTo(testCase.canvas_yaw_rad, 12);
      expect(angles.pitchRad).toBeCloseTo(testCase.canvas_pitch_rad, 12);
    }
  });

  it("bounds tracking, suspends on manual override, and re-centers exactly", () => {
    const initial: CameraState = {
      presetId: "camera.view.isometric",
      faceOnSide: "right",
      targetM: [0, 0, 0],
      zoom: 2.5,
      yawRad: 0,
      pitchRad: 0,
      trackingEnabled: false,
      trackingSuspended: false,
      autoFitEnabled: false,
    };
    const enabled = setTrackingEnabled(initial, true, [0, 0, 0]);
    const advanced = updateTrackingTarget(enabled, [10, 0, 0], 2);
    expect(advanced.targetM).toEqual([2, 0, 0]);
    expect(advanced.zoom).toBe(2.5);
    const suspended = applyManualOverride(advanced);
    expect(suspended.trackingSuspended).toBe(true);
    expect(updateTrackingTarget(suspended, [20, 0, 0], 2)).toEqual(suspended);
    const centered = recenterCamera(suspended, [20, 1, -2]);
    expect(centered.targetM).toEqual([20, 1, -2]);
    expect(centered.trackingSuspended).toBe(false);
    expect(centered.zoom).toBe(2.5);
  });

  it("rejects non-finite inputs and only reduces unsafe zoom", () => {
    expect(() => recenterCamera({
      presetId: "camera.view.isometric", faceOnSide: "right", targetM: [0, 0, 0],
      zoom: 1, yawRad: 0, pitchRad: 0,
      trackingEnabled: true, trackingSuspended: false, autoFitEnabled: false,
    }, [Number.NaN, 0, 0])).toThrow(/finite/i);
    expect(safeTrackingZoom(1.2, 0.3, 1)).toBeCloseTo(1.2);
    expect(safeTrackingZoom(4, 0.3, 1)).toBeCloseTo(2.8);
  });
});
