import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/camera_presets_v1.json";
import trackingFixture from "./__fixtures__/camera_tracking_v1.json";
import {
  AUTO_FIT_CLEARANCE_FRACTION,
  CAMERA_CONTROL_IDS,
  CAMERA_PRESET_COMMAND_IDS,
  CAMERA_TRACKING_COMMAND_IDS,
  CAMERA_TRACKING_STATE_IDS,
  TRACKING_MAX_TARGET_STEP_M,
  applyManualCameraOverride,
  applyCameraView,
  autoFitCamera,
  cameraPreset,
  canvasAngles,
  defaultCameraState,
  enforceTrackingClearance,
  recenterCamera,
  setAutoFitFallback,
  setCameraTracking,
  trackingStateId,
  updateTrackingTarget,
  validateCameraPreset,
  validateCameraState,
  type CameraViewId,
} from "./cameraPresets";

describe("canonical camera preset contract", () => {
  it("matches every shared ID and exact frame orientation", () => {
    expect(fixture.frame).toEqual({ x: "downrange", y: "up", z: "right" });
    expect(CAMERA_PRESET_COMMAND_IDS).toEqual(fixture.command_ids);
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

describe("clubhead camera tracking contract", () => {
  it("matches shared IDs, target bounds, and explicit tracking states", () => {
    expect(trackingFixture.schema).toBe("rate-of-closure-camera-tracking/v1");
    expect(CAMERA_TRACKING_COMMAND_IDS).toEqual(trackingFixture.command_ids);
    expect(CAMERA_CONTROL_IDS).toEqual(trackingFixture.control_ids);
    expect(CAMERA_TRACKING_STATE_IDS).toEqual(trackingFixture.state_ids);
    expect(TRACKING_MAX_TARGET_STEP_M).toBe(trackingFixture.maximum_target_step_m);
    expect(AUTO_FIT_CLEARANCE_FRACTION)
      .toBe(trackingFixture.minimum_clearance_fraction);
    for (const testCase of trackingFixture.target_cases) {
      const advanced = updateTrackingTarget({
        ...defaultCameraState(),
        targetM: testCase.target_m as [number, number, number],
        zoom: 2.5,
        trackingEnabled: true,
      }, testCase.subject_m as [number, number, number]);
      expect(advanced.targetM).toEqual(testCase.expected_target_m);
      expect(advanced.zoom).toBe(2.5);
    }
    const enabled = setCameraTracking(defaultCameraState(), true, [0, 0, 0]);
    expect(trackingStateId(enabled)).toBe("camera.tracking.active");
    const suspended = applyManualCameraOverride(enabled);
    expect(trackingStateId(suspended)).toBe("camera.tracking.suspended");
    expect(updateTrackingTarget(suspended, [4, 0, 0])).toEqual(suspended);
    const centered = recenterCamera(suspended, [4, 1, -2]);
    expect(centered.targetM).toEqual([4, 1, -2]);
    expect(trackingStateId(centered)).toBe("camera.tracking.active");
  });

  it("keeps the clearance fallback opt-in and reduction-only", () => {
    const initial = { ...defaultCameraState(), zoom: 1.2 };
    expect(enforceTrackingClearance(initial, 0.3, 1)).toEqual(initial);
    const safe = setAutoFitFallback(initial, true);
    expect(enforceTrackingClearance(safe, 0.3, 1)).toEqual(safe);
    const unsafe = setAutoFitFallback({ ...initial, zoom: 4 }, true);
    expect(enforceTrackingClearance(unsafe, 0.3, 1).zoom).toBeCloseTo(2.8);
  });
});
